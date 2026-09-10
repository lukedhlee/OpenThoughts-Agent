import importlib, sys
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m); print("OK  ", m, getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:120])
try:
    from skyrl_train.models import GrugMoeForCausalLM; print("OK   skyrl GrugMoeForCausalLM")
except Exception as e:
    print("FAIL skyrl GrugMoe", type(e).__name__, str(e)[:160])
