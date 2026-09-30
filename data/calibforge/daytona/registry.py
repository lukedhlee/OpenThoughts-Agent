#!/usr/bin/env python3
"""Anonymous Docker Hub registry client for the digest-pinned CalibForge images (stdlib only).

Docker Hub counts a *manifest* GET as a pull (anonymous: 100 per hour per IPv4 address or IPv6 /64, 2026-09); blob
GETs are not counted. This module caches every manifest and config it reads, so each image's manifest is fetched once
per machine. (On 2026-09-29, ~2,600 by-digest manifest GETs from one Mac never moved `ratelimit-remaining` and got no
429, but the documented limit may still apply: spread large fetches over machines or use --registry.)

    python registry.py fetch --tasks <parquet> [--recommended] --cache <dir>   # manifests + configs for the pool
    python registry.py fetch --tasks <parquet> --only <task list> --registry https://mirror.gcr.io --cache <dir>

--registry reads from a Docker Hub pull-through mirror instead, anonymously and outside Docker Hub's per-IP limit.
mirror.gcr.io answers 404 for an image it has not cached yet and fetches it in the background, so re-run until
nothing fails (a re-run also completes configs that were not cached yet). Everything is fetched by digest, so a mirror
cannot substitute content: with a non-default registry every manifest and config is checked against its sha256
before it is cached. The cache is the same either way.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REGISTRY = "https://registry-1.docker.io"
ACCEPT = ",".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
])


class Hub:
    def __init__(self, registry: str = REGISTRY):
        self.registry = registry.rstrip("/")
        self.mirror = self.registry != REGISTRY  # anonymous, no token; every byte checked against its digest
        self._tokens: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _check(self, raw: bytes, digest: str) -> bytes:
        if self.mirror and hashlib.sha256(raw).hexdigest() != digest.split(":")[1]:
            raise ValueError(f"{self.registry} served bytes that do not match {digest}")
        return raw

    def token(self, repo: str) -> str:
        with self._lock:
            tok, exp = self._tokens.get(repo, ("", 0.0))
            if time.time() < exp:
                return tok
            url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
            data = json.load(urllib.request.urlopen(url, timeout=30))
            tok = data["token"]
            self._tokens[repo] = (tok, time.time() + int(data.get("expires_in", 300)) - 30)
            return tok

    def get(self, repo: str, path: str, accept: str | None = None, tries: int = 6) -> bytes:
        for attempt in range(tries):
            auth = {} if self.mirror else {"Authorization": f"Bearer {self.token(repo)}"}
            req = urllib.request.Request(f"{self.registry}/v2/{repo}/{path}",
                                         headers={**auth, **({"Accept": accept} if accept else {})})
            try:
                return urllib.request.urlopen(req, timeout=120).read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise RuntimeError(f"{self.registry} rate limit (429) on {repo}/{path}: {e.headers}") from e
                if e.code in (401,):
                    with self._lock:
                        self._tokens.pop(repo, None)
                if e.code < 500 and e.code != 401:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):  # e.g. IncompleteRead
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GET {repo}/{path} failed after {tries} tries")

    def manifest(self, ref: str, cache: Path) -> dict:
        """ref = repo@sha256:... ; cached by digest."""
        repo, digest = ref.split("@")
        p = cache / "manifests" / f"{digest.split(':')[1]}.json"
        if p.exists():
            return json.loads(p.read_text())
        raw = self._check(self.get(repo, f"manifests/{digest}", ACCEPT), digest)
        m = json.loads(raw)
        if "manifests" in m:  # an index: pick linux/amd64
            sub = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"
                   and x.get("platform", {}).get("os") == "linux"]
            m = json.loads(self._check(self.get(repo, f"manifests/{sub[0]['digest']}", ACCEPT), sub[0]["digest"]))
            m["_index_digest"] = digest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(m))
        return m

    def config(self, repo: str, manifest: dict, cache: Path) -> dict:
        digest = manifest["config"]["digest"]
        p = cache / "configs" / f"{digest.split(':')[1]}.json"
        if p.exists():
            return json.loads(p.read_text())
        raw = self._check(self.get(repo, f"blobs/{digest}"), digest)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return json.loads(raw)

    def blob_to(self, repo: str, digest: str, dest: Path) -> Path:
        if dest.exists():
            return dest
        tmp = dest.with_name(f"{dest.name}.{threading.get_ident()}.{time.monotonic_ns()}.part")  # parallel callers
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(self.get(repo, f"blobs/{digest}"))
        tmp.replace(dest)
        return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--tasks", required=True)
    p.add_argument("--recommended", action="store_true")
    p.add_argument("--cache", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", help="file of task ids (one per line): fetch only these tasks' images")
    p.add_argument("--registry", default=REGISTRY, help="registry base URL (default Docker Hub; e.g. https://mirror.gcr.io)")
    a = ap.parse_args()
    import pandas as pd
    d = pd.read_parquet(a.tasks)
    if a.recommended:
        d = d[d.recommended_2500]
    if a.only:
        d = d[d.task_id.isin({line.strip() for line in open(a.only) if line.strip()})]
    refs = sorted(set(d.image_digest))
    if a.limit:
        refs = refs[: a.limit]
    cache = Path(a.cache)
    hub = Hub(a.registry)
    done = fail = 0

    def one(ref):
        m = hub.manifest(ref, cache)
        hub.config(ref.split("@")[0], m, cache)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, r): r for r in refs}
        for f in cf.as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print("FAIL", futs[f], e, flush=True)
                if "429" in str(e):
                    ex.shutdown(cancel_futures=True)
                    break
            if (done + fail) % 250 == 0:
                print(f"{done} ok {fail} fail", flush=True)
    print(f"done {done} ok {fail} fail of {len(refs)}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
