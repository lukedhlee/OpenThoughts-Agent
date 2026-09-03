import hashlib, json, re
_WS = re.compile(r"\s+")

def norm_ws(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return _WS.sub(" ", s).strip()

def strip_issue_markers(ps):
    if ps is None:
        return ""
    s = str(ps).strip()
    if s.startswith("[ISSUE]"):
        s = s[len("[ISSUE]"):]
    s = s.strip()
    if s.endswith("[/ISSUE]"):
        s = s[: -len("[/ISSUE]")]
    return s.strip()

def k_ps(ps):
    return hashlib.sha1(norm_ws(strip_issue_markers(ps)).encode("utf-8")).hexdigest()

def k_eo(eo):
    if eo is None:
        s = ""
    elif isinstance(eo, (bytes, bytearray)):
        s = eo.decode("utf-8", "replace")
    else:
        s = str(eo)
    try:
        obj = json.loads(s)
        canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        canon = norm_ws(s)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()
