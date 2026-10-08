async def http_get(url: str) -> str:
    return f"<html>{url}</html>"


def fetch_page(url: str) -> str:
    return http_get(url)
