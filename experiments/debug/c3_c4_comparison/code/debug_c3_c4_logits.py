"""Diagnostic comparison of C3 / C4-RAW / C4-VQ actor logits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.topology_registry import load_topologies
from experiments.train_qh_ctde import _new_env, build_ctde_trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(env, cfg, checkpoint, device):
    trainer = build_ctde_trainer(env, cfg)
    model = trainer.model
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


@torch.no_grad()
def compare_step(model_c3, model_c4, obs, valid_sta_counts, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    counts = torch.as_tensor(
        valid_sta_counts, dtype=torch.long, device=device
    ).unsqueeze(0)

    # Same raw observation -> C3 joint actor
    c3_sta, c3_mcs = model_c3.actor_logits(
        obs_t, counts, centralized=True
    )

    # Same raw observation -> C4 joint actor
    c4r_sta, c4r_mcs = model_c4.actor_logits(
        obs_t, counts, centralized=True
    )

    # Same C4 model: raw -> VQ encode -> decode -> joint actor
    _, indices, _ = model_c4.encode_observations(obs_t)
    symbols = indices.squeeze(0)
    decoded = model_c4.codec.decode(symbols)
    c4v_sta, c4v_mcs = model_c4.actor_logits(
        decoded.unsqueeze(0), counts, centralized=True
    )

    def action(sta, mcs):
        return torch.stack(
            [sta.argmax(-1), mcs.argmax(-1)], dim=-1
        ).squeeze(0).cpu().numpy()

    c3_sta_np = c3_sta.squeeze(0).cpu().numpy()
    c3_mcs_np = c3_mcs.squeeze(0).cpu().numpy()
    c4r_sta_np = c4r_sta.squeeze(0).cpu().numpy()
    c4r_mcs_np = c4r_mcs.squeeze(0).cpu().numpy()
    c4v_sta_np = c4v_sta.squeeze(0).cpu().numpy()
    c4v_mcs_np = c4v_mcs.squeeze(0).cpu().numpy()
    decoded_np = decoded.cpu().numpy()

    return {
        "raw_stats": stats(obs),
        "decoded_stats": stats(decoded_np),
        "vq_mse": float(torch.mean((decoded - obs_t.squeeze(0)) ** 2).item()),
        "vq_indices": symbols.cpu().numpy().astype(int).tolist(),
        "c3_action": action(c3_sta, c3_mcs).tolist(),
        "c4_raw_action": action(c4r_sta, c4r_mcs).tolist(),
        "c4_vq_action": action(c4v_sta, c4v_mcs).tolist(),
        "c3_sta_logits": c3_sta_np.tolist(),
        "c3_mcs_logits": c3_mcs_np.tolist(),
        "c4_raw_sta_logits": c4r_sta_np.tolist(),
        "c4_raw_mcs_logits": c4r_mcs_np.tolist(),
        "c4_vq_sta_logits": c4v_sta_np.tolist(),
        "c4_vq_mcs_logits": c4v_mcs_np.tolist(),
        "max_c3_c4raw_sta": float(np.max(np.abs(c3_sta_np - c4r_sta_np))),
        "max_c3_c4raw_mcs": float(np.max(np.abs(c3_mcs_np - c4r_mcs_np))),
        "max_c4raw_c4vq_sta": float(np.max(np.abs(c4r_sta_np - c4v_sta_np))),
        "max_c4raw_c4vq_mcs": float(np.max(np.abs(c4r_mcs_np - c4v_mcs_np))),
        "max_c3_c4vq_sta": float(np.max(np.abs(c3_sta_np - c4v_sta_np))),
        "max_c3_c4vq_mcs": float(np.max(np.abs(c3_mcs_np - c4v_mcs_np))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--topology", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--c3-checkpoint", required=True)
    p.add_argument("--c4-checkpoint", required=True)
    p.add_argument("--output-dir", default="debug_logits")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cpu")

    cfg = load_config(args.config)
    experiment = ExperimentConfig.from_dict(cfg)
    # Load selected topology
    topologies = load_topologies(args.topology, experiment.topology_file)
    topo = next(
        t for t in topologies
        if t.metadata.get("id", t.name) == args.topology
    )

    env = _new_env(topo, cfg, args.steps, seed_offset=args.seed + 10000)
    model_c3 = load_model(env, cfg, args.c3_checkpoint, device)
    model_c4 = load_model(env, cfg, args.c4_checkpoint, device)

    valid_sta_counts = [len(stas) for stas in topo.sta_positions]
    obs, _ = env.reset(seed=args.seed + 10000)

    records = []
    rows = []

    print("=" * 100)
    print("C3 / C4-RAW / C4-VQ LOGITS COMPARISON")
    print("=" * 100)

    for step in range(args.steps):
        rec = compare_step(
            model_c3, model_c4, obs, valid_sta_counts, device
        )
        rec["step"] = step
        records.append(rec)

        for ap in range(len(valid_sta_counts)):
            rows.append({
                "step": step,
                "ap": ap,
                "vq_mse": rec["vq_mse"],
                "vq_index": rec["vq_indices"][ap],
                "c3_sta_action": rec["c3_action"][ap][0],
                "c3_mcs_action": rec["c3_action"][ap][1],
                "c4_raw_sta_action": rec["c4_raw_action"][ap][0],
                "c4_raw_mcs_action": rec["c4_raw_action"][ap][1],
                "c4_vq_sta_action": rec["c4_vq_action"][ap][0],
                "c4_vq_mcs_action": rec["c4_vq_action"][ap][1],
                "c3_vs_c4raw_mcs_maxdiff": rec["max_c3_c4raw_mcs"],
                "c4raw_vs_c4vq_mcs_maxdiff": rec["max_c4raw_c4vq_mcs"],
                "c3_vs_c4vq_mcs_maxdiff": rec["max_c3_c4vq_mcs"],
            })

        print(
            f"step={step:02d} "
            f"VQ-MSE={rec['vq_mse']:.6f} | "
            f"C3={rec['c3_action']} | "
            f"C4-RAW={rec['c4_raw_action']} | "
            f"C4-VQ={rec['c4_vq_action']} | "
            f"MCS Δ(C3,C4RAW)={rec['max_c3_c4raw_mcs']:.6f} | "
            f"MCS Δ(C4RAW,C4VQ)={rec['max_c4raw_c4vq_mcs']:.6f}"
        )

        # All three policies see the same state trajectory.
        # C3 is used only to advance the environment.
        action = np.asarray(rec["c3_action"], dtype=np.int32).reshape(-1)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    csv_path = os.path.join(args.output_dir, "logits_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "logits_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    summary = {
        "steps_measured": len(records),
        "mean_vq_mse": float(np.mean([r["vq_mse"] for r in records])),
        "max_vq_mse": float(np.max([r["vq_mse"] for r in records])),
        "mean_c3_vs_c4raw_mcs_maxdiff": float(
            np.mean([r["max_c3_c4raw_mcs"] for r in records])
        ),
        "max_c3_vs_c4raw_mcs_maxdiff": float(
            np.max([r["max_c3_c4raw_mcs"] for r in records])
        ),
        "mean_c4raw_vs_c4vq_mcs_maxdiff": float(
            np.mean([r["max_c4raw_c4vq_mcs"] for r in records])
        ),
        "max_c4raw_vs_c4vq_mcs_maxdiff": float(
            np.max([r["max_c4raw_c4vq_mcs"] for r in records])
        ),
        "diagnosis": (
            "C3 and C4-RAW already differ -> inspect C4 actor/checkpoint/training"
            if np.max([r["max_c3_c4raw_mcs"] for r in records]) > 1e-6
            else
            "C3 and C4-RAW match; inspect VQ path if C4-VQ differs"
        ),
    }

    summary_path = os.path.join(args.output_dir, "logits_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"SUM : {summary_path}")


if __name__ == "__main__":
    main()
