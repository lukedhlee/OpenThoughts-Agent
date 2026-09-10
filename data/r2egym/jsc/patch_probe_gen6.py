import re
p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"
s = open(p).read()
assert "--max-model-len" not in s
s = s.replace(
    'ap.add_argument("--summarize", action="store_true"',
    'ap.add_argument("--max-model-len", type=int, default=32768, help="vLLM max_model_len; >32768 adds the grug hf_overrides (RoPE table is sized from max_seq_len, a frozen copy of max_position_embeddings)")\nap.add_argument("--summarize", action="store_true"', 1)
s = s.replace('assert a.max_in + a.max_out <= 32768, (a.max_in, a.max_out)',
              'assert a.max_in + a.max_out <= a.max_model_len, (a.max_in, a.max_out, a.max_model_len)', 1)
s = s.replace('"++generator.engine_init_kwargs.max_model_len=32768",',
              'f"++generator.engine_init_kwargs.max_model_len={a.max_model_len}",', 1)
s = s.replace('if a.summarize: args =',
              'if a.max_model_len > 32768: args.append(f"++generator.engine_init_kwargs.hf_overrides={{max_position_embeddings:{a.max_model_len},max_seq_len:{a.max_model_len}}}")\nif a.summarize: args =', 1)
assert s.count("a.max_model_len") == 6, s.count("a.max_model_len")
open(p, "w").write(s); print("patched")
