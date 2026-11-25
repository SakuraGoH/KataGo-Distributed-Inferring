"""Is the dist-server value shifted as if the position/komi differed?"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT, start_distserver  # noqa: E402

NODE = "weste"


def cfg_text(dist, extra=""):
    lines = [
        "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
        "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
        "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
        "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
        "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
    ]
    if dist:
        lines.append("distServers = 127.0.0.1:%d" % REMOTE_DS_PORT)
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


def run(c, exe, cfg, komi):
    script = "boardsize 19\nkomi %s\nkata-raw-nn b 0\nquit\n" % komi
    sftp = c.open_sftp()
    with sftp.open("/root/kp.cfg", "w") as f:
        f.write(cfg)
    with sftp.open("/root/kp.gtp", "w") as f:
        f.write(script)
    sftp.close()
    rc, out = sshutil.run(
        c, "cd /root/cpp && %s gtp -model /root/model.bin.gz -config /root/kp.cfg "
           "< /root/kp.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % exe, timeout=300)
    vals = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2:
            vals[p[0]] = p[1]
    return vals


c = sshutil.connect(NODE)
start_distserver(NODE).close()
try:
    for komi in ("6.5", "0.5", "16.5"):
        a = run(c, "./build-cuda/katago", cfg_text(False), komi)
        b = run(c, "./build-dist/katago", cfg_text(True), komi)
        print("komi %-4s direct %s   dist %s" % (komi, a, b), flush=True)
    for opt in ("0.0", "1.0"):
        a = run(c, "./build-cuda/katago", cfg_text(False, "rootPolicyOptimism = %s" % opt), "6.5")
        b = run(c, "./build-dist/katago", cfg_text(True, "rootPolicyOptimism = %s" % opt), "6.5")
        print("optimism %-4s direct %s   dist %s" % (opt, a, b), flush=True)
finally:
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()
