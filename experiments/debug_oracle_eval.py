"""
experiments/debug_oracle_eval.py
debug_c4_execution.py와 같은 환경 생성(new_env), 같은 eval_seed, 같은 env.step 경로에서
모델 없이 오라클/고정 MCS 정책을 실행한다. C4의 throughput 0이 환경/평가 경로 문제인지
학습 부족인지를 가르는 용도다.

Usage:
    python experiments/debug_oracle_eval.py --topology T3 --seed 1 --steps 20 --policy oracle
    python experiments/debug_oracle_eval.py --topology T3 --seed 1 --steps 20 --policy fixed_mcs --fixed-mcs 15
    python experiments/debug_oracle_eval.py --topology T3 --seed 1 --steps 20 --policy fixed_mcs --fixed-mcs 0

비교 방법: 같은 --topology/--seed/--steps로 debug_c4_execution.py를 돌린 Total throughput과
이 스크립트의 oracle Total throughput을 나란히 본다.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.envs.topology_registry import load_topologies
from src.training.oracle_policy import oracle_actions, prospective_harsh
from experiments.train_ablation import new_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topology", default="T3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--policy", default="oracle", choices=["oracle", "oracle_queued", "fixed_mcs"])
    parser.add_argument("--fixed-mcs", type=int, default=0)
    parser.add_argument("--output", default="debug_oracle/oracle_eval.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_seed = args.seed + 10000   # debug_c4_execution.py와 동일

    topo = load_topologies([args.topology], cfg.get("topology_file", "configs/topologies.yaml"))[0]
    env = new_env(topo, cfg, args.steps, seed_offset=eval_seed)
    obs, _ = env.reset(seed=eval_seed)

    # 계측 전의 원본 SINR 함수를 보관한다 (계획 단계 호출이 계측 기록에 섞이지 않도록).
    planning_compute = env.interference.compute

    records = []
    original_parse = env._parse_actions

    def traced_parse(actions):
        targets, mcs = original_parse(actions)
        records[-1]["parsed_targets"] = list(targets)
        records[-1]["parsed_mcs"] = list(mcs)
        return targets, mcs

    sinr_rows = []
    original_compute = env.interference.compute

    def traced_compute(*a, **kw):
        result = original_compute(*a, **kw)
        ap_idx = kw.get("ap_idx", a[0] if len(a) > 0 else None)
        sta_idx = kw.get("sta_idx", a[1] if len(a) > 1 else None)
        mcs_idx = records[-1]["parsed_mcs"][int(ap_idx)]
        sinr_rows.append({
            "step": len(records) - 1, "ap_idx": ap_idx, "sta_idx": sta_idx,
            "sinr_db": float(result.sinr_db), "mcs": int(mcs_idx),
            "required_sinr_db": float(env.mcs_table.required_sinr_db(mcs_idx)),
            "decodable": bool(env.mcs_table.decodable(result.sinr_db, mcs_idx)),
            "harsh": bool(env.is_harsh_interference),
        })
        return result

    env._parse_actions = traced_parse
    env.interference.compute = traced_compute

    total_thp, total_blocked = 0.0, 0
    thp_normal, thp_harsh = 0.0, 0.0
    try:
        for step in range(args.steps):
            harsh = prospective_harsh(env)
            action = oracle_actions(env, args.policy, args.fixed_mcs, compute_fn=planning_compute)
            records.append({"step": step, "harsh": harsh, "actions": action.tolist()})
            _, _, done, _, info = env.step(action)

            rec = records[-1]
            rec["thp"] = info.get("thp", [])
            rec["system_throughput"] = float(info.get("system_throughput", 0.0))
            rec["blocked"] = int(info.get("overlap_blocked_step", 0))
            rec["queue_sum_after_step"] = [int(sum(ap.queues)) for ap in env._aps]

            total_thp += rec["system_throughput"]
            total_blocked += rec["blocked"]
            if harsh:
                thp_harsh += rec["system_throughput"]
            else:
                thp_normal += rec["system_throughput"]
            if done:
                break
    finally:
        env.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})

    decodable = sum(r["decodable"] for r in sinr_rows)
    undecodable_at_mcs0 = sum(
        1 for r in sinr_rows if r["sinr_db"] < env.mcs_table.required_sinr_db(0)
    )
    print(f"Policy: {args.policy}" + (f" (mcs={args.fixed_mcs})" if args.policy == "fixed_mcs" else ""))
    print(f"Topology: {args.topology}; eval_seed={eval_seed}; steps={len(records)}")
    print(f"Total throughput: {total_thp}  (normal={thp_normal}, harsh={thp_harsh})")
    print(f"Total blocked: {total_blocked}")
    print(f"Decodable at selected MCS: {decodable}/{len(sinr_rows)}")
    print(f"Links with SINR below the MCS0 threshold (undecodable at any MCS): {undecodable_at_mcs0}")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
