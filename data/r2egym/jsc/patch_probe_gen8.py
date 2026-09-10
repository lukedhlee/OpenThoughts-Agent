import re
p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"
s = open(p).read()
assert "--engines" not in s
s = s.replace('ap.add_argument("--parity", action="store_true"',
 'ap.add_argument("--nodes", type=int, default=6, help="sbatch nodes: 4 policy + engines (each DP4 engine = 1 node)")\nap.add_argument("--engines", type=int, default=2, help="generator.num_inference_engines (TP1xDP4xEP4 each); nodes must be 4 + engines")\nap.add_argument("--parity", action="store_true"', 1)
s = s.replace('assert a.max_in + a.max_out <= a.max_model_len, (a.max_in, a.max_out, a.max_model_len)',
 'assert a.max_in + a.max_out <= a.max_model_len, (a.max_in, a.max_out, a.max_model_len)\nassert a.nodes == 4 + a.engines, (a.nodes, a.engines)', 1)
s = s.replace('"generator.inference_engine_expert_parallel_size=4", "generator.num_inference_engines=2",',
 '"generator.inference_engine_expert_parallel_size=4", f"generator.num_inference_engines={a.engines}",', 1)
s = s.replace('sb = re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={a.wall}", sb, flags=re.M)',
 'sb = re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={a.wall}", sb, flags=re.M)\nsb = re.sub(r"^#SBATCH --nodes=.*$", f"#SBATCH --nodes={a.nodes}", sb, flags=re.M)\nsb = sb.replace("export NUM_INFERENCE_ENGINES=24", f"export NUM_INFERENCE_ENGINES={a.nodes*4}").replace("export POLICY_NUM_NODES=6", f"export POLICY_NUM_NODES={a.nodes}")', 1)
assert s.count("a.engines") >= 2 and s.count("a.nodes") >= 5, (s.count("a.engines"), s.count("a.nodes"))
open(p, "w").write(s); print("patched")
