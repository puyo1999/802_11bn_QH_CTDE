import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.topology_registry import load_topologies
from experiments.train_ablation import (
    new_env,
    build_trainer,
    CONDITIONS,
)
from src.training.centralized_execution import CentralizedExecutor
from src.training.communication_delay import ObservationDelayChannel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topology", default="T3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument('--force-mcs', type=int, choices=range(16), default=None, help='Override MCS for every AP')
    parser.add_argument(
        "--checkpoint",
        default="experiments/debug/smoke_tests/smoke_C4_initfix/T3_C4_seed1_model.pt",
    )
    parser.add_argument(
        "--output",
        default="debug_c4_execution/c4_execution.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = ExperimentConfig.from_dict(cfg)
    train_cfg = exp.train
    agent_cfg = exp.agent
    delay_cfg = exp.communication_delay
    eval_seed = args.seed + 10000

    topologies = load_topologies(
        [args.topology],
        cfg.get("topology_file", "configs/topologies.yaml"),
    )
    topo = topologies[0]

    env = new_env(
        topo, cfg, args.steps, seed_offset=eval_seed
    )
    trainer = build_trainer(env, cfg, "C4")
    model = trainer.model

    checkpoint = torch.load(
        args.checkpoint,
        map_location=trainer.device,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=True)
    model.to(trainer.device)
    model.eval()

    valid_counts = [len(stas) for stas in topo.sta_positions]
    executor = CentralizedExecutor(
        model,
        valid_counts,
        threshold=agent_cfg.gamma_margin,
        device=train_cfg.device,
        execution_mode=CONDITIONS["C4"]["execution_mode"],
        use_age_feature=delay_cfg.enabled,
    )

    obs, _ = env.reset(seed=eval_seed)
    delay_channel = None
    if delay_cfg.enabled:
        delay_channel = ObservationDelayChannel(
            delay_cfg, len(valid_counts)
        )
        delay_channel.reset(obs, seed=eval_seed + 50000)

    # Observe the environment's actual action parsing without changing
    # the environment implementation.
    records = []
    original_parse = env._parse_actions

    def traced_parse(actions):
        parsed_targets, parsed_mcs = original_parse(actions)
        action_values = np.asarray(actions).reshape(-1).copy()
        queue_snapshot = [
            [int(q) for q in ap.queues]
            for ap in env._aps
        ]
        real_sta_counts = [
            len(env.topo.sta_positions[i])
            for i in range(env.n_aps)
        ]
        records.append({
            "step": len(records),
            "actions": action_values.tolist(),
            "queues_at_parse": queue_snapshot,
            "real_sta_counts": real_sta_counts,
            "parsed_targets": list(parsed_targets),
            "parsed_mcs": list(parsed_mcs),
        })
        return parsed_targets, parsed_mcs

    env._parse_actions = traced_parse

    # Diagnostic-only SINR tracing.
    sinr_rows = []
    original_compute = env.interference.compute

    def traced_compute(*args, **kwargs):
        result = original_compute(*args, **kwargs)
        ap_idx = kwargs.get("ap_idx", args[0] if len(args) > 0 else None)
        sta_idx = kwargs.get("sta_idx", args[1] if len(args) > 1 else None)

        row = {
            "step": len(records) - 1,
            "ap_idx": ap_idx,
            "sta_idx": sta_idx,
            "sinr_db": float(result.sinr_db),
            "blocked_by_overlap": bool(result.blocked_by_overlap),
        }

        if ap_idx is not None and 0 <= int(ap_idx) < len(records[-1]["parsed_mcs"]):
            mcs_idx = int(records[-1]["parsed_mcs"][int(ap_idx)])
            row["mcs"] = mcs_idx
            row["required_sinr_db"] = float(
                env.mcs_table.required_sinr_db(mcs_idx)
            )
            row["decodable"] = bool(
                env.mcs_table.decodable(result.sinr_db, mcs_idx)
            )

        sinr_rows.append(row)
        return result

    env.interference.compute = traced_compute


    total_throughput = 0.0
    total_blocked = 0
    try:
        for step in range(args.steps):
            delayed_obs = age_steps = None
            if delay_channel is not None:
                delay_channel.transmit(obs, step)
                delayed_obs, age_steps = delay_channel.receive(step)

            with torch.no_grad():
                decision = executor.act(
                    obs,
                    deterministic=True,
                    delayed_observations=delayed_obs,
                    observation_age_steps=age_steps,
                )

            action = np.asarray(
                decision.actions, dtype=np.int32
            ).reshape(-1)
            if args.force_mcs is not None:
                action[1::2] = args.force_mcs

            obs, _, done, _, info = env.step(action)

            rec = records[-1]
            rec["actions"] = action.tolist()
            rec["mode"] = (
                "centralized"
                if decision.centralized_policy
                else "decentralized"
            )
            rec["backhaul_bits"] = int(decision.transmitted_bits)
            rec["thp"] = info.get("thp", [])
            rec["system_throughput"] = float(
                info.get("system_throughput", 0.0)
            )
            rec["blocked"] = int(
                info.get("overlap_blocked_step", 0)
            )
            rec["queue_mean_after_step"] = info.get("queue_mean", [])
            rec["queue_sum_after_step"] = [
                int(sum(ap.queues)) for ap in env._aps
            ]

            total_throughput += rec["system_throughput"]
            total_blocked += rec["blocked"]

            if done:
                break
    finally:
        env.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        for row in records:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            })


    if sinr_rows:
        values = [r["sinr_db"] for r in sinr_rows]
        checked = [r for r in sinr_rows if "decodable" in r]
        print(f"SINR measurements: {len(values)}")
        print(
            f"SINR min/mean/max: {min(values):.2f} / "
            f"{sum(values)/len(values):.2f} / {max(values):.2f} dB"
        )
        if checked:
            passed = sum(r["decodable"] for r in checked)
            print(f"Decodable at selected MCS: {passed}/{len(checked)}")
        print("First 12 SINR records:")
        for row in sinr_rows[:12]:
            print(row)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Topology: {args.topology}; eval_seed={eval_seed}")
    print(f"Steps measured: {len(records)}")
    print(f"Total throughput: {total_throughput}")
    print(f"Total blocked: {total_blocked}")
    print(f"Backhaul bits/step: "
          f"{executor.total_transmitted_bits / max(len(records), 1):.1f}")
    print(f"CSV: {args.output}")

    print("\n--- First 5 steps ---")
    for row in records[:5]:
        print(json.dumps(row, ensure_ascii=False))

    valid = 0
    invalid = 0
    nonempty = 0
    for row in records:
        for i, target in enumerate(row["parsed_targets"]):
            if target < 0:
                invalid += 1
            else:
                valid += 1
                q = row["queues_at_parse"][i]
                if target < len(q) and q[target] > 0:
                    nonempty += 1

    print("\n--- Action parsing summary ---")
    print(f"Parsed valid STA actions: {valid}")
    print(f"Parsed invalid/empty STA actions: {invalid}")
    print(f"Valid selections with queue > 0 at parse: {nonempty}")


if __name__ == "__main__":
    main()



