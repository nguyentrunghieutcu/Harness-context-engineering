def sanitize(text: str, escape_xml: bool = True) -> str:
    if not text:
        return ""
    # simple prompt injection protection
    bad_phrases = [
        "ignore previous instructions",
        "system prompt",
        "tool override"]
    for bp in bad_phrases:
        text = text.replace(bp, "[REDACTED]")
    if escape_xml:
        text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text


def sanitize_chunks(chunks: list):
    for c in chunks:
        c.content = sanitize(c.content, escape_xml=False)
        if c.summary:
            c.summary = sanitize(c.summary, escape_xml=False)
