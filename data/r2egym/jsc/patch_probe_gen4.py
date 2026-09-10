p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"; s = open(p).read()
assert "--max-in" not in s
s = s.replace('ap.add_argument("--wall", default="05:00:00"); ap.add_argument("--lr", default="1e-5")',
  'ap.add_argument("--wall", default="05:00:00"); ap.add_argument("--lr", default="1e-5")\n'
  'ap.add_argument("--max-in", type=int, default=24576, help="max input tokens (context budget for the harness)")\n'
  'ap.add_argument("--max-out", type=int, default=8192, help="max output tokens per turn; max_in + max_out must be <= 32768")\n'
  'ap.add_argument("--summarize", action="store_true", help="enable terminus-2 context summarization")')
s = s.replace('a = ap.parse_args()\n', 'a = ap.parse_args()\nassert a.max_in + a.max_out <= 32768, (a.max_in, a.max_out)\n', 1)
s = s.replace('"++generator.engine_init_kwargs.max_model_len=32768", "generator.max_input_length=24576", "trainer.max_prompt_length=24576",\n'
              '    "++terminal_bench_config.model_info.max_input_tokens=24576", "generator.sampling_params.max_generate_length=8192",\n'
              '    "++terminal_bench_config.model_info.max_output_tokens=8192",',
              '"++generator.engine_init_kwargs.max_model_len=32768", f"generator.max_input_length={a.max_in}", f"trainer.max_prompt_length={a.max_in}",\n'
              '    f"++terminal_bench_config.model_info.max_input_tokens={a.max_in}", f"generator.sampling_params.max_generate_length={a.max_out}",\n'
              '    f"++terminal_bench_config.model_info.max_output_tokens={a.max_out}",')
s = s.replace('if a.attn_backend: args.append', 'if a.summarize: args = [x for x in args if not x.startswith("++terminal_bench_config.agent.enable_summarize=")] + ["++terminal_bench_config.agent.enable_summarize=true"]\nif a.attn_backend: args.append')
open(p, "w").write(s); print("generator patched: --max-in/--max-out/--summarize")
