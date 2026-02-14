from __future__ import annotations

import sys
from pathlib import Path


def import_mojito_module():
    try:
        import mojito as module  # type: ignore

        if hasattr(module, "KoreaInvestment") and hasattr(module, "KoreaInvestmentWS"):
            return module
    except Exception:
        pass

    local_vendor = Path(__file__).resolve().parent.parent / "mojito"
    if local_vendor.exists():
        sys.path.insert(0, str(local_vendor))
        if "mojito" in sys.modules:
            del sys.modules["mojito"]
        import mojito as module  # type: ignore

        if hasattr(module, "KoreaInvestment") and hasattr(module, "KoreaInvestmentWS"):
            return module

    raise RuntimeError("Cannot import mojito. Install mojito2 or keep ./mojito vendored.")
