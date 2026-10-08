def analyze_stream(gen):
    c = len(list(gen))
    s = sum(list(gen))
    return c, s
