def build_lookup(items: list[tuple[str, str]]) -> dict[str, str]:
    lookup = {}
    for k, v in items:
        lookup[k] = v.upper()
    return lookup
