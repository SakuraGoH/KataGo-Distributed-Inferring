"""Does the handle's max-batch size change the numbers?"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT, start_distserver  # noqa: E402

NODE = "weste"
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def run(c, exe, extra):
    sftp = c.open_sftp()
    with sftp.open("/root/mb.cfg", "w") as f:
        f.write("\n".join(BASE + extra) + "\n")
    with sftp.open("/root/mb.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s gtp -model /root/model.bin.gz -config /root/mb.cfg "
                             "< /root/mb.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % exe, timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


c = sshutil.connect(NODE)
for mb in (1, 8, 32):
    start_distserver(NODE).close()
    d = run(c, "./build-dist/katago", ["nnMaxBatchSize = %d" % mb,
                                       "distServers = 127.0.0.1:%d" % REMOTE_DS_PORT])
    e = run(c, "./build-cuda/katago", ["nnMaxBatchSize = %d" % mb])
    print("nnMaxBatchSize=%-2d  dist: %s   direct: %s" % (mb, d, e), flush=True)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
