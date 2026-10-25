"""Run inside a sandbox. Reports booleans about what the sandbox exposes; prints no secret values."""
import json, os, sys
canary = sys.argv[1]          # the fake token's value, to test for presence only
canary_path = sys.argv[2]     # host path of the fake token file
res = {}
res["canary_in_env"] = any(canary in v for v in os.environ.values())
res["canary_file_visible"] = os.path.exists(canary_path)
try:
    open("/usr/a13_write_test", "w").close(); os.remove("/usr/a13_write_test"); res["system_dir_writable"] = True
except OSError:
    res["system_dir_writable"] = False
try:
    res["net_interfaces"] = sorted(os.listdir("/sys/class/net"))
except OSError:
    res["net_interfaces"] = "unreadable"
for key, f in [("pids_max", "/sys/fs/cgroup/pids.max"), ("memory_max", "/sys/fs/cgroup/memory.max")]:
    try:
        res[key] = open(f).read().strip()
    except OSError:
        res[key] = "n/a"
res["uid"] = os.getuid()
print(json.dumps(res))
