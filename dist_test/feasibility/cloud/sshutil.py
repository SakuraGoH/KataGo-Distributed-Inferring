"""Minimal SSH/SFTP helper for the cloud verification run (paramiko).

Node credentials live in nodes.local.json next to this file (gitignored):

    {"weste": {"host": "…", "port": 22, "password": "…"}}

Only the host/port/password are needed; the user is root.
"""
import json
import sys
import time
from pathlib import Path

import paramiko

NODES_FILE = Path(__file__).parent / "nodes.local.json"


def load_nodes():
    if not NODES_FILE.exists():
        return {}
    raw = json.loads(NODES_FILE.read_text(encoding="utf-8"))
    return {name: (n["host"], int(n["port"]), n.get("password", ""))
            for name, n in raw.items()}


HOSTS = load_nodes()


def connect(name, timeout=30):
    host, port, password = HOSTS[name]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username="root", password=password,
              timeout=timeout, allow_agent=False, look_for_keys=False,
              banner_timeout=30, auth_timeout=30)
    t = c.get_transport()
    t.set_keepalive(20)
    return c


def run(c, cmd, timeout=180, check=False):
    chan = c.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out, err = [], []
    while True:
        if chan.recv_ready():
            out.append(chan.recv(65536))
        if chan.recv_stderr_ready():
            err.append(chan.recv_stderr(65536))
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.02)
    rc = chan.recv_exit_status()
    text = (b"".join(out) + b"".join(err)).decode("utf-8", "replace")
    chan.close()
    if check and rc != 0:
        raise RuntimeError("command failed rc=%d: %s\n%s" % (rc, cmd, text[-2000:]))
    return rc, text


def run_stream(c, cmd, logpath, poll=1.0):
    """Run a long command, stream output to a local log; returns exit code."""
    chan = c.get_transport().open_session()
    chan.exec_command(cmd)
    logf = open(logpath, "wb")
    try:
        while True:
            got = False
            while chan.recv_ready():
                d = chan.recv(65536)
                logf.write(d)
                logf.flush()
                got = True
            while chan.recv_stderr_ready():
                d = chan.recv_stderr(65536)
                logf.write(d)
                logf.flush()
                got = True
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
            if not got:
                time.sleep(poll)
        rc = chan.recv_exit_status()
    finally:
        logf.close()
        chan.close()
    return rc


def put(c, local, remote, mkdir_parents=True):
    sftp = c.open_sftp()
    try:
        if mkdir_parents:
            parts = Path(remote).parent.as_posix()
            if parts and parts != "/":
                _, _ = run(c, "mkdir -p '%s'" % parts)
        sftp.put(str(local), remote)
    finally:
        sftp.close()


def main():
    name = sys.argv[1]
    cmd = sys.argv[2]
    c = connect(name)
    rc, out = run(c, cmd, timeout=300)
    print(out, end="")
    print("[exit %d]" % rc)
    c.close()


if __name__ == "__main__":
    main()
