p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"
s = open(p).read()
assert "--parity" not in s
s = s.replace('ap.add_argument("--summarize", action="store_true"',
 'ap.add_argument("--parity", action="store_true", help="SFT-format parity: skip_special_tokens=false via harbor extra_body (think markers kept in history; needs harbor lukedhlee/terminus2-think-parity) + chat_template_content_format=string (needs MarinSkyRL lukedhlee/jupiter-parity64k)")\nap.add_argument("--summarize", action="store_true"', 1)
s = s.replace('if a.summarize: args =',
 'if a.parity: args += ["++terminal_bench_config.harbor.extra_body={skip_special_tokens:false}", "++generator.engine_init_kwargs.chat_template_content_format=string"]\nif a.summarize: args =', 1)
assert s.count("a.parity") == 1 and "--parity" in s
open(p, "w").write(s); print("patched")
