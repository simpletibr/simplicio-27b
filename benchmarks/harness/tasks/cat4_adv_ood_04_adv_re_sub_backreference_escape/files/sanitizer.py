import re
def redact_phones(text: str) -> str:
    return re.sub(r'(\d{3})-(\d{4})', '[REDACTED]', text)
