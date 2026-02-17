from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    cleaned = text.replace(",", "").replace("%", "").replace("+", "")
    if cleaned in {"-", "--", "."}:
        return None

    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match is None:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def normalize_code(raw: Any) -> str:
    text = str(raw).strip()
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)
    return text


def normalize_symbol(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    text = text.replace("/", ".")
    match = re.search(r"([A-Z][A-Z0-9\.\-_]{0,14})", text)
    if match:
        return match.group(1)
    return text


def pick_first(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return None


def normalize_yyyymmdd(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 8:
        return digits
    return None


def latest_list_of_dict(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    preferred = ("output2", "output", "output1", "data", "result")
    for key in preferred:
        value = data.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return value

    list_values = [
        value
        for value in data.values()
        if isinstance(value, list) and (not value or isinstance(value[0], dict))
    ]
    if not list_values:
        return []
    return max(list_values, key=len)


def first_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("output", "output1", "data", "result"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    for value in data.values():
        if isinstance(value, dict):
            return value
    return {}


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def maintained(
    values: List[float],
    threshold: float,
    min_samples: int,
    min_ratio: float,
) -> Tuple[bool, Optional[float], Optional[float]]:
    if len(values) < min_samples:
        return False, mean(values), None
    hit_ratio = sum(v >= threshold for v in values) / len(values)
    return hit_ratio >= min_ratio, mean(values), hit_ratio


def parse_universe_file(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse a universe file.

    Supported line formats:
    - 005930
    - 005930,삼성전자
    - 005930 삼성전자
    - ...any text containing a 6-digit code (first code is used)
    """
    codes: List[str] = []
    names: Dict[str, str] = {}
    seen = set()

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        code = ""
        name = ""

        match_sep = re.match(r"^\s*(\d{6})\s*[,|\t]\s*(.+?)\s*$", line)
        if match_sep:
            code = match_sep.group(1)
            name = match_sep.group(2).strip()
        else:
            match_space = re.match(r"^\s*(\d{6})\s+(.+?)\s*$", line)
            if match_space:
                code = match_space.group(1)
                name = match_space.group(2).strip()
            else:
                codes_in_line = re.findall(r"\b\d{6}\b", line)
                if not codes_in_line:
                    continue
                code = codes_in_line[0]

        if code in seen:
            if name and not names.get(code):
                names[code] = name
            continue

        seen.add(code)
        codes.append(code)
        if name:
            names[code] = name

    return codes, names


def parse_us_universe_file(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse a US stock universe file.

    Supported line formats:
    - AAPL
    - AAPL,Apple Inc.
    - AAPL Apple Inc.
    """
    symbols: List[str] = []
    names: Dict[str, str] = {}
    seen = set()

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        symbol = ""
        name = ""

        match_sep = re.match(r"^\s*([A-Za-z][A-Za-z0-9\.\-_]{0,14})\s*[,|\t]\s*(.+?)\s*$", line)
        if match_sep:
            symbol = normalize_symbol(match_sep.group(1))
            name = match_sep.group(2).strip()
        else:
            match_space = re.match(r"^\s*([A-Za-z][A-Za-z0-9\.\-_]{0,14})\s+(.+?)\s*$", line)
            if match_space:
                symbol = normalize_symbol(match_space.group(1))
                name = match_space.group(2).strip()
            else:
                match_symbol = re.search(r"\b[A-Za-z][A-Za-z0-9\.\-_]{0,14}\b", line)
                if not match_symbol:
                    continue
                symbol = normalize_symbol(match_symbol.group(0))

        if not symbol:
            continue

        if symbol in seen:
            if name and not names.get(symbol):
                names[symbol] = name
            continue

        seen.add(symbol)
        symbols.append(symbol)
        if name:
            names[symbol] = name

    return symbols, names


def extract_codes_and_names_from_df(df: Any, max_count: int) -> Tuple[List[str], Dict[str, str]]:
    if df is None or getattr(df, "empty", True):
        return [], {}

    sample = df.head(min(len(df.index), 500))
    best_col = None
    best_ratio = -1.0
    for col in sample.columns:
        series = sample[col].astype(str).str.strip()
        ratio = series.str.fullmatch(r"\d{6}").mean()
        if ratio > best_ratio:
            best_ratio = float(ratio)
            best_col = col

    if best_col is None:
        return [], {}

    name_col = None
    for col in sample.columns:
        if col == best_col:
            continue
        series = sample[col].astype(str).str.strip()
        non_numeric_ratio = (~series.str.fullmatch(r"[\d\.\-]+")).mean()
        if float(non_numeric_ratio) > 0.5:
            name_col = col
            break

    codes: List[str] = []
    names: Dict[str, str] = {}
    seen = set()
    for _, row in df.iterrows():
        code = normalize_code(row.get(best_col))
        if not re.fullmatch(r"\d{6}", code):
            continue
        if code in seen:
            continue
        seen.add(code)
        codes.append(code)
        if name_col is not None:
            names[code] = str(row.get(name_col, "")).strip()
        if len(codes) >= max_count:
            break

    return codes, names


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"
