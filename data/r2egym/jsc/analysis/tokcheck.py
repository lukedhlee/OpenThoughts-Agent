import json
from transformers import AutoTokenizer
M="/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok=AutoTokenizer.from_pretrained(M)
for s in ["<think>","</think>","<|start_think|>","<|end_think|>","<think>\n","\n</think>\n"]:
    print(repr(s), "->", tok.encode(s, add_special_tokens=False))
print("decode 128002..128003:", [tok.decode([i]) for i in (128002,128003)])
print("decode 128000..128012:", [(i,tok.decode([i])) for i in range(128000,128013)])
print("all_special_ids has 128002/128003:", 128002 in tok.all_special_ids, 128003 in tok.all_special_ids)
tj=json.load(open(M+"/tokenizer.json"))
at=[a for a in tj.get("added_tokens",[]) if a["id"] in (128002,128003) or "think" in a["content"].lower()]
print("tokenizer.json added tokens w/ think:", at)
print("n added tokens:", len(tj.get("added_tokens",[])), "ids range", min(a["id"] for a in tj["added_tokens"]), max(a["id"] for a in tj["added_tokens"]))
msgs=[{"role":"user","content":"hi"}]
print("gen prompt:", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
print("gen prompt enable_thinking=True:", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)))
msgs2=[{"role":"user","content":"hi"},{"role":"assistant","content":"reasoning text\n{\"a\":1}"},{"role":"user","content":"obs"}]
print("history render:", repr(tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)))
print("template overhead per user+assistant pair (tokens):", len(tok.apply_chat_template(msgs2, tokenize=True, add_generation_prompt=True)) - len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)) - len(tok.encode("reasoning text\n{\"a\":1}",add_special_tokens=False)) - len(tok.encode("obs",add_special_tokens=False)))
