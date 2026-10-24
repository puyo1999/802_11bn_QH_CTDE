"""Diagnostic comparison of C3 (raw joint) and C4 (VQ joint) policies.

The script evaluates already-trained C3/C4 checkpoints on identical T3/T6
initial states and records:
  1) original observation statistics,
  2) C4 VQ reconstruction statistics and MSE,
  3) deterministic C3 action,
  4) C4 action using VQ-decoded observations,
  5) C4 action using the SAME model with raw observations,
  6) per-step throughput/blocking/backhaul.

The C4 raw-input branch is the key control: if C4-RAW behaves normally while
C4-VQ collapses, the fault is localized to the VQ transport/reconstruction
path rather than the C4 actor itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.space_validator import build_adjacency, validate_spaces
from src.envs.topology_registry import load_topologies
from src.envs.env_factory import make_env
from experiments.train_ablation import AblationActorCritic, CONDITIONS


def make_env_for_topology(topo, cfg, horizon, seed):
    experiment = ExperimentConfig.from_dict(cfg)
    env_config = replace(experiment.env, max_episodes=horizon, seed=seed)
    env = make_env(
        env_config, topology=topo, seed_offset=0,
        topology_file=experiment.topology_file,
    )
    space = validate_spaces(
        env, topo, declared_state_dim=experiment.agent.state_dim, strict=True
    )
    env.space_spec = space
    return env


def build_model(env, cfg, condition, device):
    experiment = ExperimentConfig.from_dict(cfg)
    agent_cfg = experiment.agent
    delay_enabled = experiment.communication_delay.enabled
    spec = CONDITIONS[condition]
    model = AblationActorCritic(
        observation_dim=env.space_spec.local_obs_dim + int(delay_enabled),
        n_agents=env.space_spec.n_aps,
        n_stas=env.space_spec.action_dims_per_ap[0],
        n_mcs=env.space_spec.action_dims_per_ap[1],
        hidden_dim=agent_cfg.hidden_dim,
        vq_bits=agent_cfg.vq_bits,
        adjacency=torch.as_tensor(build_adjacency(env.topo), dtype=torch.bool),
        critic_type=spec["critic"], actor_type=spec["actor"],
    ).to(device)
    return model


def load_model(env, cfg, condition, checkpoint, device):
    model = build_model(env, cfg, condition, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def stats(x: torch.Tensor) -> dict[str, float]:
    a = x.detach().float().cpu().numpy()
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def deterministic_action(model, obs_np, valid_counts, device):
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    counts = torch.as_tensor(valid_counts, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        sta_logits, mcs_logits = model.actor_logits(
            obs, counts, centralized=True
        )
        sta = sta_logits.argmax(-1)
        mcs = mcs_logits.argmax(-1)
        action = torch.stack((sta, mcs), dim=-1).squeeze(0)
        sta_prob = torch.softmax(sta_logits, dim=-1).max(-1).values.squeeze(0)
        mcs_prob = torch.softmax(mcs_logits, dim=-1).max(-1).values.squeeze(0)
    return (
        action.cpu().numpy().astype(np.int32),
        sta_logits.squeeze(0).cpu(), mcs_logits.squeeze(0).cpu(),
        sta_prob.cpu().numpy(), mcs_prob.cpu().numpy(),
    )


def decode_c4(model, obs_np, device):
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        quantized, indices, codec_loss = model.encode_observations(obs)
        # ObservationCodec.decode() accepts the symbol tensor returned by encode().
        decoded = model.codec.decode(indices)
    return obs.squeeze(0), quantized.squeeze(0), indices.squeeze(0), decoded.squeeze(0), codec_loss


def count_invalid(actions, valid_counts):
    sta = actions[:, 0]
    return int(sum(int(sta[i]) >= int(valid_counts[i]) for i in range(len(valid_counts))))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qh_ctde.yaml")
    p.add_argument("--topology", default="T3")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--c3-checkpoint", required=True)
    p.add_argument("--c4-checkpoint", required=True)
    p.add_argument("--output-dir", default="debug_C3_C4")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    experiment = ExperimentConfig.from_dict(cfg)
    device = args.device or experiment.train.device
    eval_seed = args.seed + 10_000

    topologies = load_topologies(
        [args.topology], cfg.get("topology_file", "configs/topologies.yaml")
    )
    if not topologies:
        raise RuntimeError(f"Topology not found: {args.topology}")
    topo = topologies[0]

    # Three identical environment trajectories:
    # C3 raw, C4 VQ, and C4 raw-control.
    env_c3 = make_env_for_topology(topo, cfg, args.steps, eval_seed)
    env_c4_vq = make_env_for_topology(topo, cfg, args.steps, eval_seed)
    env_c4_raw = make_env_for_topology(topo, cfg, args.steps, eval_seed)

    try:
        model_c3 = load_model(env_c3, cfg, "C3", args.c3_checkpoint, device)
        model_c4 = load_model(env_c4_vq, cfg, "C4", args.c4_checkpoint, device)
        valid_counts = [len(stas) for stas in topo.sta_positions]

        obs_c3, _ = env_c3.reset(seed=eval_seed)
        obs_c4, _ = env_c4_vq.reset(seed=eval_seed)
        obs_raw, _ = env_c4_raw.reset(seed=eval_seed)
        np.testing.assert_allclose(obs_c3, obs_c4, rtol=0, atol=0,
                                   err_msg="C3/C4 initial observations differ")
        np.testing.assert_allclose(obs_c3, obs_raw, rtol=0, atol=0,
                                   err_msg="C4 raw-control initial observation differs")

        rows = []
        summary = {
            "topology": args.topology, "seed": args.seed,
            "evaluation_seed": eval_seed, "steps": args.steps,
            "device": str(device),
            "valid_sta_counts": valid_counts,
        }
        totals = {"c3": 0.0, "c4_vq": 0.0, "c4_raw": 0.0,
                  "c3_blocked": 0, "c4_vq_blocked": 0, "c4_raw_blocked": 0}
        mse_values = []

        for step in range(args.steps):
            # The three branches share the same initial state. After step 0 their
            # trajectories may legitimately diverge because the policies choose
            # different actions; that divergence is itself part of the diagnostic.

            obs_t, quantized_t, indices_t, decoded_t, codec_loss = decode_c4(
                model_c4, obs_c4, device
            )
            recon_error = float(torch.mean((obs_t - decoded_t) ** 2).item())
            mse_values.append(recon_error)

            a3, s3, m3, sp3, mp3 = deterministic_action(
                model_c3, obs_c3, valid_counts, device
            )
            a4vq, s4vq, m4vq, sp4vq, mp4vq = deterministic_action(
                model_c4, decoded_t.cpu().numpy(), valid_counts, device
            )
            a4raw, s4raw, m4raw, sp4raw, mp4raw = deterministic_action(
                model_c4, obs_raw, valid_counts, device
            )

            next3, _, done3, _, info3 = env_c3.step(a3.reshape(-1))
            next4, _, done4, _, info4 = env_c4_vq.step(a4vq.reshape(-1))
            nextr, _, doner, _, infor = env_c4_raw.step(a4raw.reshape(-1))

            t3 = float(info3.get("system_throughput", 0.0))
            t4 = float(info4.get("system_throughput", 0.0))
            tr = float(infor.get("system_throughput", 0.0))
            b3 = int(info3.get("overlap_blocked_step", 0))
            b4 = int(info4.get("overlap_blocked_step", 0))
            br = int(infor.get("overlap_blocked_step", 0))
            totals["c3"] += t3; totals["c4_vq"] += t4; totals["c4_raw"] += tr
            totals["c3_blocked"] += b3; totals["c4_vq_blocked"] += b4; totals["c4_raw_blocked"] += br

            row = {
                "step": step,
                "obs_min": stats(obs_t)["min"],
                "obs_max": stats(obs_t)["max"],
                "obs_mean": stats(obs_t)["mean"],
                "obs_std": stats(obs_t)["std"],
                "decoded_min": stats(decoded_t)["min"],
                "decoded_max": stats(decoded_t)["max"],
                "decoded_mean": stats(decoded_t)["mean"],
                "decoded_std": stats(decoded_t)["std"],
                "vq_mse": recon_error,
                "vq_indices": json.dumps(indices_t.detach().cpu().reshape(-1).tolist()),
                "codec_loss": float(codec_loss.item()) if torch.is_tensor(codec_loss) else float(codec_loss),
                "c3_actions": json.dumps(a3.tolist()),
                "c4_vq_actions": json.dumps(a4vq.tolist()),
                "c4_raw_actions": json.dumps(a4raw.tolist()),
                "c3_invalid_sta": count_invalid(a3, valid_counts),
                "c4_vq_invalid_sta": count_invalid(a4vq, valid_counts),
                "c4_raw_invalid_sta": count_invalid(a4raw, valid_counts),
                "c3_sta_conf_mean": float(np.mean(sp3)),
                "c4_vq_sta_conf_mean": float(np.mean(sp4vq)),
                "c4_raw_sta_conf_mean": float(np.mean(sp4raw)),
                "c3_mcs_conf_mean": float(np.mean(mp3)),
                "c4_vq_mcs_conf_mean": float(np.mean(mp4vq)),
                "c4_raw_mcs_conf_mean": float(np.mean(mp4raw)),
                "c3_throughput": t3,
                "c4_vq_throughput": t4,
                "c4_raw_throughput": tr,
                "c3_blocked": b3,
                "c4_vq_blocked": b4,
                "c4_raw_blocked": br,
                "c3_backhaul_bits": 0,
                "c4_vq_backhaul_bits": int(indices_t.numel() * model_c4.codec.bits_per_observation),
                "c4_raw_backhaul_bits": 0,
            }
            rows.append(row)

            obs_c3, obs_c4, obs_raw = next3, next4, nextr
            if done3 or done4 or doner:
                break

        n = len(rows)
        summary.update({
            "actual_steps": n,
            "c3_throughput": totals["c3"],
            "c4_vq_throughput": totals["c4_vq"],
            "c4_raw_throughput": totals["c4_raw"],
            "c3_throughput_per_step": totals["c3"] / max(n, 1),
            "c4_vq_throughput_per_step": totals["c4_vq"] / max(n, 1),
            "c4_raw_throughput_per_step": totals["c4_raw"] / max(n, 1),
            "c3_blocked_per_step": totals["c3_blocked"] / max(n, 1),
            "c4_vq_blocked_per_step": totals["c4_vq_blocked"] / max(n, 1),
            "c4_raw_blocked_per_step": totals["c4_raw_blocked"] / max(n, 1),
            "vq_mse_mean": float(np.mean(mse_values)) if mse_values else None,
            "vq_mse_max": float(np.max(mse_values)) if mse_values else None,
            "c4_vq_backhaul_bits_per_step": float(np.mean([r["c4_vq_backhaul_bits"] for r in rows])) if rows else 0.0,
        })

        os.makedirs(args.output_dir, exist_ok=True)
        write_csv(os.path.join(args.output_dir, "c3_c4_step_diagnostics.csv"), rows)
        with open(os.path.join(args.output_dir, "c3_c4_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))
        print(f"\nWrote: {os.path.join(args.output_dir, 'c3_c4_step_diagnostics.csv')}")
        print(f"Wrote: {os.path.join(args.output_dir, 'c3_c4_summary.json')}")

    finally:
        env_c3.close(); env_c4_vq.close(); env_c4_raw.close()


if __name__ == "__main__":
    main()
