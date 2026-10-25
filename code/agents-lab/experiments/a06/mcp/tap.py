"""Run an MCP stdio server and log every line that crosses its stdin/stdout (a wire tap).

usage: python tap.py LOGFILE -- COMMAND ARGS...
"""
import json, subprocess, sys, threading, time

log_path, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
log = open(log_path, "a")
lock = threading.Lock()
child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def record(direction, line):
    with lock:
        log.write(json.dumps({"t": time.time(), "dir": direction, "msg": line.decode().rstrip("\n")}) + "\n")
        log.flush()


def pump(src, dst, direction):
    for line in iter(src.readline, b""):
        record(direction, line)
        dst.write(line)
        dst.flush()
    dst.close()


up = threading.Thread(target=pump, args=(sys.stdin.buffer, child.stdin, "client->server"), daemon=True)
up.start()
pump(child.stdout, sys.stdout.buffer, "server->client")
child.wait()
