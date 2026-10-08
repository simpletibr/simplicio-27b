import re
def is_valid_slug(slug: str) -> bool:
    return bool(re.match(r'^[a-z0-9-]+$', slug))
