"""Cap on older teacher reasoning in the student's view (Luke 2026-09-25 17:40 PT), shared by the router and the SFT
renderer so both produce the same text.

In the student's history, the most recent teacher turn keeps its reasoning in full; every older teacher turn's
reasoning is cut to its first CAP_TOKENS tokens of the 09-21 tokenizer, at the last sentence or line boundary before
that point, with no marker. The teacher's own view is unchanged.

    cut, at = cut_reasoning(text, tokenizer)   # at = char offset of the cut, or None when not cut
"""
import os
import re

os.environ.setdefault('RAYON_NUM_THREADS', '1')        # login node: 4,096-pid cap, tokenizers spawns a pool per core
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

CAP_TOKENS = 1000
BOUNDARY_RE = re.compile(r'(?<=[.!?])\s|\n')


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def cut_reasoning(text, tok, cap=CAP_TOKENS):
    """(text cut at a sentence/line boundary within its first `cap` tokens, char offset) or (text, None)."""
    if not text:
        return text, None
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) <= cap:
        return text, None
    limit = enc.offsets[cap - 1][1]
    b = None
    for m in BOUNDARY_RE.finditer(text, 0, limit + 1):
        b = m.start() if text[m.start()] == '\n' else m.start()
    if b is None or b < limit // 2:                    # no boundary in the second half: last whitespace instead
        ws = text.rfind(' ', 0, limit)
        b = ws if ws > 0 else limit
    return text[:b].rstrip(), b


def teacher_turn_for_student(content, reasoning, tok=None, cut=False, cap=CAP_TOKENS):
    """A teacher turn as the student sees it: <|start_think|>{reasoning}<|end_think|>{content}, stripped, no newlines
    around the span; reasoning cut when `cut` (an older teacher turn) and a tokenizer is given. Returns
    (text, cut_at or None, reasoning used)."""
    r = (reasoning or '').strip()
    at = None
    if cut and tok is not None and r:
        r, at = cut_reasoning(r, tok, cap)
    return (f'<|start_think|>{r}<|end_think|>' if r else '') + (content or '').strip(), at, r
