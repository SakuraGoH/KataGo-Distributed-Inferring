"""Test the production-style path: OpenSSH `-L` instead of the paramiko tunnel.

Also re-checks raw-NN values, to tell a transport problem apart from a
protocol/server problem.
"""
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CLOUD = Path(__file__).resolve().parent
sys.path.insert(0, str(CLOUD))
sys.path.insert(0, r"C:\Users\10632\Desktop\SakuraLink")
import sshutil  # noqa: E402
from cloud_test import (KATAGO_CLIENT, KATAGO_LOCAL, MODEL, REMOTE_DS_PORT,  # noqa: E402
                        make_cfg, gtp, raw_text, bench)

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
KEY = CLOUD / "id_ed25519"
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
LOCAL_PORTS = {"weste": 21731, "westc": 21732}
NODES = ("weste", "westc")


def ensure_key():
    if not KEY.exists():
        subprocess.run([SSH.replace("ssh.exe", "ssh-keygen.exe"), "-t", "ed25519",
                        "-N", "", "-f", str(KEY), "-q"], check=True)
    return (KEY.with_suffix(".pub")).read_text().strip()


def install_key(name, pub):
    c = sshutil.connect(name)
    sshutil.run(c, "mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys")
    rc, out = sshutil.run(c, "grep -qF '%s' /root/.ssh/authorized_keys && echo yes || echo no" % pub.split()[1])
    if "yes" not in out:
        sshutil.run(c, "echo '%s' >> /root/.ssh/authorized_keys" % pub)
    c.close()


def wait_port(port, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=1)
            s.close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sshl_"))
    pub = ensure_key()
    procs = []
    try:
        for h in NODES:
            install_key(h, pub)
            host, port, _ = sshutil.HOSTS[h]
            p = subprocess.Popen(
                [SSH, "-N", "-i", str(KEY), "-p", str(port),
                 "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                 "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                 "-L", "127.0.0.1:%d:127.0.0.1:%d" % (LOCAL_PORTS[h], REMOTE_DS_PORT),
                 "root@" + host],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            procs.append((h, p))
            ok = wait_port(LOCAL_PORTS[h])
            print("ssh -L %s :%d up=%s" % (h, LOCAL_PORTS[h], ok), flush=True)
            if not ok:
                print(p.stderr.read().decode("utf-8", "replace")[-500:])

        # raw-NN through ssh -L
        cfg = make_cfg(tmp / "sshl.cfg", "127.0.0.1:%d" % LOCAL_PORTS["weste"], 1, batch=1)
        r = gtp(KATAGO_CLIENT, cfg, SCRIPT, timeout=120)
        text = raw_text(r.stdout)
        win = [ln.split()[1] for ln in text.splitlines() if ln.startswith("whiteWin")]
        lead = [ln.split()[1] for ln in text.splitlines() if ln.startswith("whiteLead")]
        print("ssh -L raw-nn: whiteWin=%s whiteLead=%s (rc=%s)" % (win, lead, r.returncode))

        # throughput through ssh -L
        for tag, servers, threads in (
                ("weste only", "127.0.0.1:%d" % LOCAL_PORTS["weste"], 1),
                ("both 2conn", ",".join("127.0.0.1:%d" % LOCAL_PORTS[h] for h in NODES), 2),
                ("both 4conn", ",".join("127.0.0.1:%d" % LOCAL_PORTS[h] for h in NODES), 4)):
            for batch in (8, 32):
                cfg = make_cfg(tmp / ("b%d_%s.cfg" % (batch, tag.replace(" ", "_"))),
                               servers, threads, batch=batch)
                b = bench(cfg, batch, 20 if batch == 8 else 10)
                print("  %-10s batch %-2d: %8.1f evals/s  %7.2f ms/batch"
                      % (tag, batch, b["evals_s"], b["ms_per_batch"]), flush=True)
    finally:
        for h, p in procs:
            p.terminate()
        for h, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
