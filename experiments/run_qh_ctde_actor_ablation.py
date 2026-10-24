"""Run paired multi-seed actor ablations on selected QH-CTDE topologies.

Example:
    python experiments/run_qh_ctde_actor_ablation.py --topologies T3 T6 --seeds 11 22 33 44 55

Each condition uses the same centralized critic and PPO settings. Actor modes
are local, forced joint execution with raw or VQ-compressed observations, and
the coherence-gated auto mode. CSV tables and PNG figures are written to the
configured output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import dump_config, load_config
from src.config.schema import ExperimentConfig
from src.envs.topology_registry import load_topologies
from src.training.centralized_execution import CentralizedExecutor
from src.training.communication_delay import ObservationDelayChannel
from experiments.train_qh_ctde import _new_env, train_qh_ctde


VARIANT_LABELS = {
    "local": "Local actor",
    "joint_raw": "Joint actor (raw)",
    "joint_vq": "Joint actor (VQ)",
    "auto": "Auto gate",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, topo, cfg: dict, seed: int, execution_mode: str,
             test_steps: int) -> dict[str, float | int | str]:
    """Evaluate a fixed execution mode and collect scheduling/backhaul metrics."""
    eval_seed = seed + 10_000
    env = _new_env(topo, cfg, test_steps, seed_offset=eval_seed)
    train_cfg = ExperimentConfig.from_dict(cfg).train
    agent_cfg = ExperimentConfig.from_dict(cfg).agent
    delay_config = ExperimentConfig.from_dict(cfg).communication_delay
    executor = CentralizedExecutor(
        model,
        [len(stas) for stas in topo.sta_positions],
        threshold=agent_cfg.gamma_margin,
        device=train_cfg.device,
        execution_mode=execution_mode,
        use_age_feature=delay_config.enabled,
    )
    obs, _ = env.reset(seed=eval_seed)
    delay_channel = None
    if delay_config.enabled:
        delay_channel = ObservationDelayChannel(delay_config, len(topo.sta_positions))
        delay_channel.reset(obs, seed=eval_seed + 50_000)
    throughput = 0.0
    blocked_events = 0
    steps = 0
    try:
        for _ in range(test_steps):
            delayed_obs = age_steps = None
            if delay_channel is not None:
                request_central = (
                    execution_mode in {"joint_raw", "joint_vq"}
                    or (execution_mode == "auto"
                        and float(obs[0, -1]) <= agent_cfg.gamma_margin)
                )
                if request_central:
                    delay_channel.transmit(obs, steps)
                delayed_obs, age_steps = delay_channel.receive(steps)
            decision = executor.act(
                obs, deterministic=True,
                delayed_observations=delayed_obs,
                observation_age_steps=age_steps,
            )
            action = np.asarray(decision.actions, dtype=np.int32).reshape(-1)
            obs, _, done, _, info = env.step(action)
            throughput += float(info.get("system_throughput", 0.0))
            blocked_events += int(info.get("overlap_blocked_step", 0))
            steps += 1
            if done:
                break
    finally:
        env.close()
    return {
        "topology": topo.metadata.get("id", topo.name),
        "variant": execution_mode,
        "seed": seed,
        "evaluation_seed": eval_seed,
        "throughput_packets": throughput,
        "blocked_events": blocked_events,
        "evaluation_steps": steps,
        "blocked_events_per_step": blocked_events / max(steps, 1),
        "backhaul_bits": executor.total_transmitted_bits,
        "backhaul_bits_per_step": executor.total_transmitted_bits / max(steps, 1),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_training_summary(rows: list[dict]) -> list[dict]:
    """Summarize reward histories per seed and across paired seeds."""
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["topology"], row["variant"], int(row["seed"]))].append(
            float(row["reward"]))
    per_seed = []
    for (topology, variant, seed), rewards in sorted(grouped.items()):
        values = np.asarray(rewards, dtype=np.float64)
        per_seed.append({
            "level": "seed", "topology": topology, "variant": variant,
            "seed": seed, "n_seeds": 1,
            "tail10_mean": float(values[-10:].mean()),
            "tail20_mean": float(values[-20:].mean()),
            "tail20_std_within_run": float(values[-20:].std()),
            "tail20_std_across_seeds": "",
            "auc_mean": float(values.mean()),
        })
    conditions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_seed:
        conditions[(row["topology"], row["variant"])].append(row)
    aggregate = []
    for (topology, variant), runs in sorted(conditions.items()):
        tail20 = np.asarray([run["tail20_mean"] for run in runs])
        aggregate.append({
            "level": "aggregate", "topology": topology, "variant": variant,
            "seed": "", "n_seeds": len(runs),
            "tail10_mean": float(np.mean([r["tail10_mean"] for r in runs])),
            "tail20_mean": float(tail20.mean()),
            "tail20_std_within_run": "",
            # Across-seed SD is calculated over per-seed tail-window means.
            "tail20_std_across_seeds": float(tail20.std(ddof=1)) if len(tail20) > 1 else 0.0,
            "auc_mean": float(np.mean([r["auc_mean"] for r in runs])),
        })
    return per_seed + aggregate


def build_evaluation_summary(rows: list[dict]) -> list[dict]:
    """Aggregate evaluation metrics by topology and actor variant."""
    metrics = ("throughput_packets", "blocked_events_per_step", "backhaul_bits_per_step")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["topology"], row["variant"])].append(row)
    summary = []
    for (topology, variant), runs in sorted(groups.items()):
        entry = {"topology": topology, "variant": variant, "n_seeds": len(runs)}
        for metric in metrics:
            vals = np.asarray([float(run[metric]) for run in runs])
            entry[f"{metric}_mean"] = float(vals.mean())
            entry[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(entry)
    return summary


def build_overlap_summary(rows: list[dict]) -> list[dict]:
    """Aggregate branch-specific training blocked events across episodes/seeds."""
    per_seed: dict[tuple[str, str, int, str], dict[str, int]] = defaultdict(
        lambda: {"steps": 0, "blocked_events": 0})
    for row in rows:
        key = (row["topology"], row["variant"], int(row["seed"]), row["mode"])
        per_seed[key]["steps"] += int(row["steps"])
        per_seed[key]["blocked_events"] += int(row["blocked_events"])
    conditions: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for (topology, variant, seed, mode), values in sorted(per_seed.items()):
        conditions[(topology, variant, mode)].append({"seed": seed, **values})
    summary = []
    for (topology, variant, mode), runs in sorted(conditions.items()):
        total_steps = sum(run["steps"] for run in runs)
        total_blocked = sum(run["blocked_events"] for run in runs)
        per_seed_rates = [run["blocked_events"] / run["steps"]
                          for run in runs if run["steps"] > 0]
        summary.append({
            "topology": topology, "variant": variant, "mode": mode,
            "n_seeds": len(runs), "total_steps": total_steps,
            "total_blocked_events": total_blocked,
            "blocked_events_per_step": (total_blocked / total_steps if total_steps else ""),
            "mean_seed_blocked_events_per_step": (
                float(np.mean(per_seed_rates)) if per_seed_rates else ""),
            "std_seed_blocked_events_per_step": (
                float(np.std(per_seed_rates, ddof=1)) if len(per_seed_rates) > 1
                else (0.0 if per_seed_rates else "")),
        })
    return summary


def plot_learning_curves(histories: dict[tuple[str, str], list[np.ndarray]],
                         output_dir: str, smooth_window: int = 10) -> None:
    for topology in sorted({key[0] for key in histories}):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for variant in VARIANT_LABELS:
            runs = histories.get((topology, variant), [])
            if not runs:
                continue
            length = min(map(len, runs))
            values = np.stack([run[:length] for run in runs])
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            if smooth_window > 1 and length >= smooth_window:
                kernel = np.ones(smooth_window, dtype=float) / smooth_window
                mean = np.convolve(mean, kernel, mode="valid")
                spread = np.convolve(spread, kernel, mode="valid")
                x = np.arange(smooth_window - 1, length)
            else:
                x = np.arange(length)
            ax.plot(x, mean, linewidth=2, label=VARIANT_LABELS[variant])
            ax.fill_between(x, mean - spread, mean + spread, alpha=0.16)
        ax.set(title=f"{topology}: actor ablation learning curves",
               xlabel="Episode", ylabel="Episode reward")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{topology}_reward_curves.png"), dpi=180)
        plt.close(fig)


def plot_evaluation(evaluation_rows: list[dict], output_dir: str) -> None:
    metrics = [
        ("throughput_packets", "Evaluation throughput (packets)", "throughput"),
        ("blocked_events_per_step", "Overlap blocked events / step", "overlap_block_rate"),
        ("backhaul_bits_per_step", "Backhaul bits / step", "backhaul_bits"),
    ]
    topologies = sorted({row["topology"] for row in evaluation_rows})
    for topology in topologies:
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
        for ax, (key, title, _) in zip(axes, metrics):
            means, errors = [], []
            for variant in VARIANT_LABELS:
                values = [float(row[key]) for row in evaluation_rows
                          if row["topology"] == topology and row["variant"] == variant]
                means.append(float(np.mean(values)) if values else 0.0)
                errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            ax.bar(range(len(VARIANT_LABELS)), means, yerr=errors, capsize=4,
                   color=["#607D8B", "#E69F00", "#009E73", "#CC79A7"])
            ax.set_xticks(range(len(VARIANT_LABELS)), [
                {"local": "Local", "joint_raw": "Joint\nraw",
                 "joint_vq": "Joint\nVQ", "auto": "Auto\ngate"}[variant]
                for variant in VARIANT_LABELS
            ])
            ax.set_title(title)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.suptitle(f"{topology}: deterministic evaluation (mean ± SD across seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{topology}_evaluation_metrics.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topologies", nargs="+", default=["T3", "T6"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_LABELS),
                        default=list(VARIANT_LABELS))
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override train.max_episodes for a smoke run")
    parser.add_argument("--rollout-length", type=int, default=None,
                        help="Override train.rollout_length for a smoke run")
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="Override train.eval_steps")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--csv-dir", default=None,
                        help="CSV output directory (default: <output-dir>/csv)")
    parser.add_argument("--plot-dir", default=None,
                        help="PNG output directory (default: <output-dir>/plots)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["train"]["max_episodes"] = args.episodes
    if args.rollout_length is not None:
        cfg["train"]["rollout_length"] = args.rollout_length
    if args.eval_steps is not None:
        cfg["train"]["eval_steps"] = args.eval_steps
    if cfg["train"]["max_episodes"] < 1 or cfg["train"]["rollout_length"] < 1:
        raise ValueError("max_episodes and rollout_length must be positive")
    experiment = ExperimentConfig.from_dict(cfg)
    default_run_name = ("actor_ablation_delay_v2" if experiment.communication_delay.enabled
                        else "actor_ablation_v2")
    output_dir = args.output_dir or os.path.join(experiment.train.result_dir, default_run_name)
    csv_dir = args.csv_dir or os.path.join(output_dir, "csv")
    plot_dir = args.plot_dir or os.path.join(output_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    dump_config(cfg, os.path.join(output_dir, "resolved_config.yaml"))
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump({"topologies": args.topologies, "seeds": args.seeds,
                   "variants": args.variants, "episodes": cfg["train"]["max_episodes"],
                   "rollout_length": cfg["train"]["rollout_length"],
                   "eval_steps": cfg["train"]["eval_steps"]}, stream, indent=2)
    topologies = load_topologies(args.topologies, cfg.get("topology_file", "configs/topologies.yaml"))

    history_by_condition: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    training_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    overlap_rows: list[dict] = []
    total = len(topologies) * len(args.seeds) * len(args.variants)
    run_num = 0
    for topo in topologies:
        topology_id = topo.metadata.get("id", topo.name)
        for seed in args.seeds:
            for variant in args.variants:
                run_num += 1
                print(f"\n[{run_num}/{total}] topology={topology_id} seed={seed} actor={variant}",
                      flush=True)
                run_cfg = {**cfg, "env": {**cfg.get("env", {}), "seed": seed}}
                seed_everything(seed)
                episode_metrics: list[dict] = []
                model, history = train_qh_ctde(
                    topo, run_cfg, seed_offset=seed, execution_mode=variant,
                    episode_metrics=episode_metrics)
                history_array = np.asarray(history, dtype=np.float32)
                history_by_condition[(topology_id, variant)].append(history_array)
                training_rows.extend({
                    "topology": topology_id,
                    "variant": variant,
                    "seed": seed,
                    "episode": episode + 1,
                    "reward": float(reward),
                } for episode, reward in enumerate(history_array))
                for metric in episode_metrics:
                    for mode, key in (("centralized", "centralized"),
                                      ("decentralized", "decentralized")):
                        overlap_rows.append({
                            "topology": topology_id, "variant": variant, "seed": seed,
                            "episode": metric["episode"], "mode": mode,
                            "steps": metric[f"{key}_steps"],
                            "blocked_events": metric[f"{key}_blocked_events"],
                            "blocked_events_per_step": metric[
                                f"{key}_blocked_events_per_step"],
                        })
                evaluation_rows.append(evaluate(
                    model, topo, run_cfg, seed, variant,
                    ExperimentConfig.from_dict(run_cfg).train.eval_steps))
                # Flush after every completed run so a later interruption keeps prior results.
                write_csv(os.path.join(csv_dir, "training_rewards.csv"), training_rows)
                write_csv(os.path.join(csv_dir, "evaluation_metrics.csv"), evaluation_rows)
                write_csv(os.path.join(csv_dir, "training_overlap_by_mode.csv"), overlap_rows)
                write_csv(os.path.join(csv_dir, "training_summary.csv"),
                          build_training_summary(training_rows))
                write_csv(os.path.join(csv_dir, "evaluation_summary.csv"),
                          build_evaluation_summary(evaluation_rows))
                write_csv(os.path.join(csv_dir, "training_overlap_summary.csv"),
                          build_overlap_summary(overlap_rows))
                # Keep usable interim figures if a long multi-seed job is stopped.
                plot_learning_curves(history_by_condition, plot_dir)
                plot_evaluation(evaluation_rows, plot_dir)

    print(f"\nCSV files: {os.path.abspath(csv_dir)}")
    print(f"PNG plots: {os.path.abspath(plot_dir)}")


if __name__ == "__main__":
    main()
