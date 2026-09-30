"""The one shared TMax base on Daytona and its snapshot recipe.

Every TMax task's Dockerfile starts `FROM ubuntu:22.04`. The snapshot is that base, pinned by digest, plus harbor's
agent tooling (tmux, asciinema; appended by harbor's bake_agent_tooling at snapshot build). Everything after FROM in a
task's own Dockerfile (ENV, COPY of fixtures and install scripts, RUN of base_install.sh / post_install.sh / pip) is
replayed at sandbox start by setup_files/setup.sh, so one snapshot serves all 14,601 tasks.

The pin is jammy-20260410, the release under the one prebuilt TMax image whose base layer matches a plain ubuntu
release (hamishi740/swerl-tmax-v3:37a79d0fd9b9, layer f63eb04151bc) and the last jammy before TMax's May 2026
generation. The replay runs apt-get update itself, so the package versions a task gets are today's jammy archive
either way; the pin only fixes the handful of packages the base image ships.
"""
from __future__ import annotations

LABEL = "jammy"
IMAGE = "ubuntu@sha256:14be402d3f1eeeb5e7da73d3260322c68e7b51c88388f53e88eb21d6450bd520"  # jammy-20260410, amd64
# the base image's config Env (Docker starts every build step from it, then applies the Dockerfile's ENV lines)
BASE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}


def dockerfile() -> str:
    return f"""# TMax shared Daytona base '{LABEL}' (OpenThoughts-Agent data/tmax/daytona).
# One snapshot serves every TMax task. Each task's own Dockerfile steps after FROM (ENV, COPY, RUN) are replayed at
# sandbox start by setup_files/setup.sh, which deletes /setup_files when it is done. Nothing task-specific lives here.
FROM {IMAGE}
"""
