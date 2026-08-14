#!/usr/bin/env python3
"""Runboard bundler: inline experiment JSONs into template.html -> dist/dashboard.html.

Usage:
  python build.py                # bundles every data/*.json (sorted)
  python build.py data/a.json data/b.json   # explicit order = selector order
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def main():
    files = [pathlib.Path(p) for p in sys.argv[1:]] or sorted((HERE / "data").glob("*.json"))
    exps = [json.loads(p.read_text()) for p in files]
    template = (HERE / "template.html").read_text()
    payload = json.dumps(exps)
    if "</script" in payload.replace("<\\/script", ""):
        raise SystemExit("experiment JSON contains a </script> sequence; refusing to inline")
    html = template.replace("__EXPERIMENTS_JSON__", payload)
    out = HERE / "dist" / "dashboard.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"bundled {len(exps)} experiment(s) -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    for e in exps:
        print(f"  - {e['id']}: {e['title']} (snapshot {e.get('snapshot', '?')})")


if __name__ == "__main__":
    main()
