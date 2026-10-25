"""Part 13: sandboxed tool execution and a permission policy for the lab harness.

Two isolation backends run a command in a throwaway work directory:

  bwrap   bubblewrap: new user, pid, net, ipc, uts namespaces; read-only system dirs;
          only the work dir is writable; environment cleared.
  docker  a hardened container (prefix vidyalai-a13): --network none, --read-only root,
          --cap-drop ALL, no-new-privileges, non-root user, pids/memory/cpu limits,
          only the work dir mounted.

Secrets are never real: make_canary() writes a random fake token into the scratch dir and
the host-side environment, and the isolation checks report only whether it is visible.

Policy: every tool call is classified allow / ask / deny before it runs (deny wins, then ask,
then allow, the same order Claude Code documents). harness.py and ledger.py are imported,
not changed.
"""

import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

SCRATCH = Path("/home/amit/.cache/vidyalai-scratch/a13")
IMAGE = "vidyalai-a13-python:3.12-slim"
PREFIX = "vidyalai-a13"


def make_canary():
    """A fake secret: never a real credential. Written outside the work dir."""
    token = "CANARY-" + secrets.token_hex(12)
    d = SCRATCH / "fakehome" / ".config" / "fakeapp"
    d.mkdir(parents=True, exist_ok=True)
    (d / "token").write_text(token)
    return token, d / "token"


def new_workdir(name):
    w = SCRATCH / "work" / name
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True)
    return w


def bwrap_cmd(work, argv):
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/work",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(work), "/work", "--chdir", "/work", *argv]


def docker_flags(work, name):
    return ["--name", name, "--runtime", "runc", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--pids-limit", "64", "--memory", "512m",
            "--memory-swap", "512m", "--cpus", "1", "--tmpfs", "/tmp:size=64m",
            "-v", f"{work}:/work", "-w", "/work", "-e", "HOME=/work"]


def run_bwrap(work, argv, timeout=30):
    return subprocess.run(bwrap_cmd(work, argv), capture_output=True, text=True, timeout=timeout,
                          env={"PATH": "/usr/bin:/bin"})


def run_docker_fresh(work, argv, name, timeout=60):
    return subprocess.run(["docker", "run", "--rm", *docker_flags(work, name), IMAGE, *argv],
                          capture_output=True, text=True, timeout=timeout)


class WarmContainer:
    """One long-lived hardened container; each tool call is a docker exec into it."""
    def __init__(self, work, name):
        self.name = name
        subprocess.run(["docker", "run", "-d", "--rm", *docker_flags(work, name), IMAGE,
                        "sleep", "3600"], check=True, capture_output=True, text=True)

    def run(self, argv, timeout=30):
        return subprocess.run(["docker", "exec", self.name, *argv], capture_output=True,
                              text=True, timeout=timeout)

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return time.perf_counter() - t0, r


# ---- Permission policy ---------------------------------------------------------------

def classify(call, policy):
    """Return 'allow', 'ask' or 'deny' for one tool call {name, input} under a policy."""
    return policy(call)
