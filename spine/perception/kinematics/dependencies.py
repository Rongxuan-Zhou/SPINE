"""Helpers to validate optional external dependencies for adapters."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger(__name__)


def require_dependency(
    module_name: str, repo_name: str, repo_url: str, install_subdir: str
) -> None:
    """Raise a clear ImportError when optional adapter deps are missing."""
    if os.getenv("SPINE_SKIP_DEP_CHECK"):
        logger.debug("跳过依赖检查 %s 因为设置了 SPINE_SKIP_DEP_CHECK", module_name)
        return
    module_available = importlib.util.find_spec(module_name) is not None
    external_path = _REPO_ROOT / "external" / install_subdir
    if module_available:
        return
    hint = (
        f"{repo_name} 未安装或未在 PYTHONPATH 中。"
        f" 请将仓库 clone 到 {external_path} (来源 {repo_url}) 并执行 `pip install -e {external_path}`。"
    )
    raise ImportError(hint)


__all__ = ["require_dependency"]
