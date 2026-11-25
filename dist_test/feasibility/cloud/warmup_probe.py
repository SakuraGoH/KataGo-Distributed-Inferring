"""Is the first eval on a fresh CUDA handle different from the second?"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT, start_distserver  # noqa: E402

NODE = "weste"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]
SCRIPT = ("boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nkata-raw-nn b 0\nkata-raw-nn b 0\nquit\n")


def cfg(dist):
    lines = list(BASE)
    if dist:
        lines.append("distServers = 127.0.0.1:%d" % REMOTE_DS_PORT)
    return "\n".join(lines) + "\n"


def run(c, exe, config):
    sftp = c.open_sftp()
    with sftp.open("/root/tw.cfg", "w") as f:
        f.write(config)
    with sftp.open("/root/tw.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s gtp -model /root/model.bin.gz -config /root/tw.cfg "
                             "< /root/tw.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % exe, timeout=300)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


c = sshutil.connect(NODE)
start_distserver(NODE).close()
try:
    print("direct (3 evals on one handle):")
    for ln in run(c, "./build-cuda/katago", cfg(False)):
        print("   ", ln)
    print("dist (3 evals on one connection):")
    for ln in run(c, "./build-dist/katago", cfg(True)):
        print("   ", ln)
finally:
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()
