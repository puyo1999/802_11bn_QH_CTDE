"""
src/config/merge.py
YAML 설정 딕셔너리 병합 전용 유틸.

기존 load_config()는 base.update(cfg) 형태의 얕은 병합을 사용했다.
이 방식은 자식 YAML이 'env:' 키를 일부만 재정의해도 base.yaml의 env 섹션
전체가 소실되는 문제를 발생시킨다 (configs/qh_ctde.yaml 주석 참조).
본 모듈은 재귀적 깊은 병합(deep merge)만을 담당한다.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    base 위에 override를 재귀적으로 덮어쓴 새 딕셔너리를 반환한다.

    규칙:
      - 양쪽 값이 모두 dict이면 재귀 병합
      - 그 외(스칼라, 리스트)는 override 값으로 치환
      - 입력 딕셔너리는 변형하지 않는다 (deepcopy 기반)
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_all(*layers: dict[str, Any]) -> dict[str, Any]:
    """여러 설정 레이어를 왼쪽부터 순서대로 깊은 병합한다."""
    result: dict[str, Any] = {}
    for layer in layers:
        if layer:
            result = deep_merge(result, layer)
    return result
