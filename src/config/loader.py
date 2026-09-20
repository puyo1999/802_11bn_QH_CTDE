"""
src/config/loader.py
YAML 설정 파일 로딩 및 _base_ 상속 체인 해석.

기존 experiments/train.py, experiments/train_qh_ctde.py에 중복 정의되어 있던
load_config()를 대체한다. 시그니처는 동일하게 유지하여 기존 호출부를 보존한다.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from src.config.merge import deep_merge
from src.config.overrides import apply_overrides

_BASE_KEY = "_base_"
_MAX_DEPTH = 8


def read_yaml(path: str) -> dict[str, Any]:
    """단일 YAML 파일을 읽어 딕셔너리로 반환한다."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_bases(cfg: dict[str, Any], config_dir: str, depth: int) -> dict[str, Any]:
    """_base_ 키를 따라 상위 설정을 재귀적으로 병합한다."""
    if depth > _MAX_DEPTH:
        raise RecursionError("_base_ 상속 깊이가 한계를 초과했습니다 (순환 참조 의심)")

    base_ref = cfg.pop(_BASE_KEY, None)
    if base_ref is None:
        return cfg

    base_paths = [base_ref] if isinstance(base_ref, str) else list(base_ref)

    merged: dict[str, Any] = {}
    for rel_path in base_paths:
        base_path = rel_path if os.path.isabs(rel_path) else os.path.join(config_dir, rel_path)
        base_cfg = read_yaml(base_path)
        base_cfg = _resolve_bases(base_cfg, os.path.dirname(base_path), depth + 1)
        merged = deep_merge(merged, base_cfg)

    return deep_merge(merged, cfg)


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """
    설정 파일을 로드하고 _base_ 체인과 명령행 오버라이드를 모두 반영한다.

    Args:
        path:      진입 YAML 경로
        overrides: ['train.lr=1e-4', ...] 형태의 dot-path 할당 목록

    Returns:
        완전히 병합된 설정 딕셔너리
    """
    cfg = read_yaml(path)
    cfg = _resolve_bases(cfg, os.path.dirname(os.path.abspath(path)), depth=0)
    cfg = apply_overrides(cfg, overrides)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def dump_config(cfg: dict[str, Any], path: str) -> None:
    """병합 완료된 설정을 재현성 확보를 위해 실험 결과 디렉토리에 기록한다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
