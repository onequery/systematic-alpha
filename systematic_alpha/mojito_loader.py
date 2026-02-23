from __future__ import annotations

import sys
from pathlib import Path


def _supports_core(module) -> bool:
    return bool(hasattr(module, "KoreaInvestment") and hasattr(module, "KoreaInvestmentWS"))


def _supports_us_methods(module) -> bool:
    ki = getattr(module, "KoreaInvestment", None)
    if ki is None:
        return False
    has_price = hasattr(ki, "fetch_price")
    has_detail = hasattr(ki, "fetch_price_detail_oversea")
    has_ohlcv = hasattr(ki, "fetch_ohlcv_oversea") or hasattr(ki, "fetch_ohlcv_overesea")
    return bool(has_price and has_detail and has_ohlcv)


def import_mojito_module():
    try:
        import mojito as module  # type: ignore

        if _supports_core(module) and _supports_us_methods(module):
            return module
    except Exception:
        pass

    local_vendor = Path(__file__).resolve().parent.parent / "mojito"
    if local_vendor.exists():
        sys.path.insert(0, str(local_vendor))
        for mod_name in list(sys.modules.keys()):
            if mod_name == "mojito" or mod_name.startswith("mojito."):
                del sys.modules[mod_name]
        import mojito as module  # type: ignore

        if _supports_core(module) and _supports_us_methods(module):
            return module

    raise RuntimeError(
        "Cannot import a mojito module with required US methods. "
        "Install compatible mojito2 or keep ./mojito vendored."
    )
