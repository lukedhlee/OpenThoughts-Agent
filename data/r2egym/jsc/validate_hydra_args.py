"""validate_hydra_args.py <rl_config.json> — compose MarinSkyRL ppo_base_config with the config JSON hydra args (hydra compose API,
no skyrl/torch import; login-node safe with OMP_NUM_THREADS=1) and list every override the schema rejects. Exit 1 if any."""
import json, sys, re
from hydra import initialize_config_dir, compose
from hydra.errors import ConfigCompositionException
CONF = "/e/project1/transfernetx/lee27/code/MarinSkyRL/skyrl-train/skyrl_train/config"
args = json.load(open(sys.argv[1]))["skyrl_hydra_args"]; bad = []
with initialize_config_dir(version_base=None, config_dir=CONF):
    while True:
        try:
            compose(config_name="ppo_base_config", overrides=args); break
        except ConfigCompositionException as e:
            msg = str(e); m = re.search(r"Could not override '([^']+)'", msg) or re.search(r"Key '([^']+)' is not in struct", msg)
            key = m.group(1) if m else None
            hit = [x for x in args if key and (x.split("=")[0].lstrip("+") == key or x.split("=")[0].lstrip("+").endswith("." + key))]
            if not hit: print("UNPARSED:", msg.splitlines()[0][:300]); bad.append(msg.splitlines()[0]); break
            bad.append(hit[0]); args = [x for x in args if x != hit[0]]
print("REJECTED:", bad if bad else "none"); sys.exit(1 if bad else 0)
