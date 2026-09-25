"""The three shared CalibForge bases on Daytona and the snapshot recipe for each.

A task belongs to the base whose layers form the longest prefix of the task image's layers. The snapshot is that
exact base image (digest-pinned, so its bytes equal the task image's lower layers) plus harbor's agent tooling and
a small hidden helper dir (/opt/cfdelta: static curl, CA bundle, the tooling's dpkg stanzas). setup_files/setup.sh
removes the helper dir after applying the task's own layers, so the agent never sees it.
"""
from __future__ import annotations

# Static helpers baked into every base, pinned by sha256 (checked inside the build).
CURL_URL = "https://github.com/moparisthebest/static-curl/releases/download/v8.21.0/curl-amd64"
CURL_SHA256 = "e45ae1805ceb3f816e9768fbbaba05d98a2dc9c5577f66395f37660941d88e19"
CACERT_URL = "https://curl.se/ca/cacert-2026-08-13.pem"
CACERT_SHA256 = "f66dff1bdf8f96060b8177976f8b7d9254bc89bc4db933d769f7384d28480bc9"

# label -> base image (amd64 manifest digest) and the layer digests every covered task image starts with.
BASES = {
    "ubuntu2404": {
        "image": "ubuntu@sha256:019e8eb29a85e74d64925745884f2ec79aa27e3feab36353d24656f4d6b89467",  # noble-20260730.1
        "layers": ["sha256:966c395d29cb24a3faf7e04f32878fe5778819d4132daee4f47e2aaf7b9af924"],
        "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
    },
    "tb2verifier": {
        "image": "aweaiteam/calibforge@sha256:0f61b2776c40231d5d868278ed6042295ddbc9d24b036cd72f0b88ebe9bd621a",  # tb2-verifier-base-v1
        "layers": [
            "sha256:966c395d29cb24a3faf7e04f32878fe5778819d4132daee4f47e2aaf7b9af924",
            "sha256:92fbde8a1f488af2e8f3c58198dce83095492e3c9c8ae762c8d3431316ff6c92",
            "sha256:c09c50064db4fd84e62ff465d4034dbe53d4b8cf942f94be4e72a4aeac6980a5",
            "sha256:16f2af61f63fcd35cc11f80c2e023a8a01129993d8276812615b6c2bf1246a02",
            "sha256:47ae1ff3979d9def94a36ef95c93eba5ee08bc82995e1e98dc3356cb21de51cf",
            "sha256:7800e065aef29e3cc0be7baf55b60eb650e46b1c82535a1f7c762748b10430da",
            "sha256:7c2d18b2d820c5d6f7127ab50c5ac87c33173a27d93d9d6a6b9689fc5f02d381",
        ],
        "env": [
            "PATH=/opt/verifier/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PIP_INDEX_URL=https://pypi.org/simple", "UV_INDEX_URL=https://pypi.org/simple",
            "UV_DEFAULT_INDEX=https://pypi.org/simple", "UV_CACHE_DIR=/opt/uv/cache", "UV_PYTHON_INSTALL_DIR=/opt/uv/python",
            "UV_TOOL_DIR=/opt/uv/tools", "UV_LINK_MODE=copy", "UV_PYTHON_PREFERENCE=managed",
            "PIP_DISABLE_PIP_VERSION_CHECK=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "UV_PYTHON_DOWNLOADS=never",
        ],
    },
    "bookworm": {
        "image": "debian@sha256:362e64223cc0da95422b3b13c045186fc0a81250e765d31c025fbddf257f6143",  # bookworm-20260803-slim
        "layers": ["sha256:039e6f9f9752f74a3ff4a6a224f64c7c864da16ed98f882107704328f41b9c42"],
        "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
    },
}

# harbor's bake_agent_tooling layer, run here too (before the dpkg diff) so its stanzas can be merged back after a
# task layer replaces /var/lib/dpkg/status. bake_agent_tooling appends the same RUN again; that one is a no-op.
TOOLING_RUN = ("RUN DEBIAN_FRONTEND=noninteractive apt-get update "
               "&& DEBIAN_FRONTEND=noninteractive apt-get install -y tmux asciinema "
               "&& rm -rf /var/lib/apt/lists/*")


def dockerfile(label: str) -> str:
    b = BASES[label]
    return f"""# CalibForge shared Daytona base '{label}' (OpenThoughts-Agent data/calibforge/daytona).
# One snapshot serves every task whose image starts with this base's layers. Each task's own layers are applied at
# sandbox start by setup_files/setup.sh, which deletes /opt/cfdelta when it is done. Nothing task-specific lives here.
FROM {b['image']}
USER root
RUN mkdir -p /opt/cfdelta && cp /var/lib/dpkg/status /opt/cfdelta/status.base
{TOOLING_RUN}
RUN awk 'NR==FNR {{ if ($1 == "Package:") base[$2] = 1; next }} /^Package: / {{ keep = !($2 in base) }} keep' \\
      /opt/cfdelta/status.base /var/lib/dpkg/status > /opt/cfdelta/tooling.status && rm /opt/cfdelta/status.base
ADD {CURL_URL} /opt/cfdelta/curl
ADD {CACERT_URL} /opt/cfdelta/cacert.pem
RUN echo "{CURL_SHA256}  /opt/cfdelta/curl" | sha256sum -c - \\
 && echo "{CACERT_SHA256}  /opt/cfdelta/cacert.pem" | sha256sum -c - \\
 && chmod 0755 /opt/cfdelta/curl && chmod 0644 /opt/cfdelta/cacert.pem
WORKDIR /app
"""
