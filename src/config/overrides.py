"""
src/config/overrides.py
명령행 dot-path 오버라이드 처리.

사용 예:
    python experiments/train_qh_ctde.py --config configs/qh_ctde.yaml \
        --set train.lr=1e-4 env.topology=T3 agent.use_dsm=false
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


def parse_scalar(raw: str) -> Any:
    """
    문자열을 YAML 스칼라 규칙으로 해석한다 (int/float/bool/null/list/str).

    PyYAML은 YAML 1.1 규칙을 따르므로 '1e-4'처럼 소수점과 부호가 생략된
    지수 표기를 부동소수점으로 인식하지 않고 문자열로 반환한다.
    학습률 지정에 해당 표기가 빈번하므로 숫자 변환을 별도로 시도한다.
    """
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        parsed = raw

    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed


def set_by_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """'a.b.c' 경로에 값을 기록한다. 중간 경로가 없으면 생성한다."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def get_by_path(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """'a.b.c' 경로의 값을 읽는다. 경로가 없으면 default를 반환한다."""
    node: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def apply_overrides(cfg: dict[str, Any], assignments: list[str] | None) -> dict[str, Any]:
    """
    ['train.lr=1e-4', 'agent.use_dsm=false'] 형태의 리스트를 설정에 적용한다.
    원본 cfg는 변형하지 않는다.
    """
    if not assignments:
        return cfg

    patched = deepcopy(cfg)
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"오버라이드 형식 오류 (key=value 필요): {item}")
        dotted_key, raw_value = item.split("=", 1)
        set_by_path(patched, dotted_key.strip(), parse_scalar(raw_value.strip()))
    return patched
