import re


def find_match(pattern, text):
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None
