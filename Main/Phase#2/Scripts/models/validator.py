import re

_RULES = {
    "bmw": [
        re.compile(r'^(?P<main_group>\d{2})(?P<subgroup>\d{2})(?P<core7>\d{7})$'),
        re.compile(r'^\d{4}5A[0-9A-F]{5}$', re.IGNORECASE)
    ],
    "vag": [
        re.compile(r'^(?P<platform>[A-Z0-9]{3})(?P<group>\d{3})(?P<item>\d{3})(?P<revision>[A-Z0-9]*)$')
    ],
    "mercedes": [
        re.compile(r'^(?P<prefix>[A-Z])?(?P<core>\d{10})(?P<suffix>[A-Z0-9]*)$')
    ]
}

_MB_PREFIXES = set("ABNC")
NON_MB_PREFIXES = set("XZMLDJESTKVW")


def validate_article(article, current_label):
    article = str(article).strip()

    # TRASH
    if len(article) < 5 or not any(c.isalnum() for c in article):
        return "manual_check", "rule-override: manual (format invalid)"

    # BMW
    for pattern in _RULES["bmw"]:
        if pattern.match(article):
            return "bmw", "rule-override: bmw"
    if "MINI" in article.upper() or article.startswith("88"):
        return "bmw", "rule-override: bmw (MINI variant)"

    # VAG
    for pattern in _RULES["vag"]:
        if pattern.match(article):
            return "vag", "rule-override: vag"

    # MB
    for pattern in _RULES["mercedes"]:
        match = pattern.match(article)
        if match:
            prefix = match.group('prefix')
            if (prefix is None or prefix in _MB_PREFIXES) and prefix not in NON_MB_PREFIXES:
                return "mercedes", "rule-override: mercedes"
            continue

    if current_label != "unknown_article":
        return current_label, f"model-decision: {current_label}"

    # Keep unknowns as unknowns; manual_check should be reserved for
    # malformed inputs or explicit low-confidence policy.
    return "unknown_article", "model-decision: unknown_article"


def register_new_brand(brand_name, regex_patterns):
    _RULES[brand_name] = [re.compile(p) for p in regex_patterns]