import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from Scripts.features.atomar import (
    BMW_RE, VAG_RE, MB_PREFIXES,
    TOYOTA_GENERAL_RE, TOYOTA_SUBARU_RE,
    HONDA_GENERAL_RE, HONDA_HARDWARE_RE,
    NISSAN_GENERAL_RE, NISSAN_HARDWARE_RE,
    MITSUBISHI_CLASSIC_RE, MITSUBISHI_MODERN_RE,
)

# Brand detection rules: {label: callable(article_str) -> bool}
#
# Keys must match training brand labels ("bmw", "mercedes", "vag").
# To add a new brand: append one entry here — no other changes required.
#
# IMPORTANT: dict insertion order determines column order (Python 3.7+).
# After training, do NOT reorder or remove entries — breaks scaler alignment.
BRAND_RULES = {
    "bmw":        lambda x: bool(BMW_RE.match(str(x))),
    "mercedes":   lambda x: bool(x) and str(x)[0] in MB_PREFIXES,
    "vag":        lambda x: bool(VAG_RE.match(str(x))),
    "toyota":     lambda x: bool(TOYOTA_GENERAL_RE.match(str(x)) or TOYOTA_SUBARU_RE.match(str(x))),
    "honda":      lambda x: bool(HONDA_GENERAL_RE.match(str(x)) or HONDA_HARDWARE_RE.match(str(x))),
    "nissan":     lambda x: bool(NISSAN_GENERAL_RE.match(str(x)) or NISSAN_HARDWARE_RE.match(str(x))),
    "mitsubishi": lambda x: bool(MITSUBISHI_CLASSIC_RE.match(str(x)) or MITSUBISHI_MODERN_RE.match(str(x))),
}
