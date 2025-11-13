from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import cast


def test_slack_dump_persistent_factory_is_always_none() -> None:
    mod: ModuleType = importlib.import_module(
        "x_make_slack_dump_and_reset_z.x_cls_make_slack_dump_and_reset_x"
    )
    # Factory must be None regardless of environment
    assert getattr(mod, "PersistentEnvReaderFactory", None) is None


def test_slack_token_resolution_does_not_import_persistent_tool() -> None:
    mod: ModuleType = importlib.import_module(
        "x_make_slack_dump_and_reset_z.x_cls_make_slack_dump_and_reset_x"
    )
    target_mod = "x_make_persistent_env_var_x.x_cls_make_persistent_env_var_x"
    if target_mod in sys.modules:
        del sys.modules[target_mod]
    # Call resolver directly
    token, used = cast(
        "tuple[str | None, bool]",
        mod._resolve_persistent_slack_token(),
    )
    assert token is None
    assert used is False
    assert target_mod not in sys.modules
