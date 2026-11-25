"""Does requireExactNNLen explain the value difference?"""
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
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def run(c, exe, extra):
    lines = list(BASE) + list(extra)
    sftp = c.open_sftp()
    with sftp.open("/root/re.cfg", "w") as f:
        f.write("\n".join(lines) + "\n")
    with sftp.open("/root/re.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s gtp -model /root/model.bin.gz -config /root/re.cfg "
                             "< /root/re.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % exe, timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


c = sshutil.connect(NODE)
start_distserver(NODE).close()
try:
    print("direct requireMaxBoardSize=true :", run(c, "./build-cuda/katago", ["requireMaxBoardSize = true"]))
    print("direct requireMaxBoardSize=false:", run(c, "./build-cuda/katago", ["requireMaxBoardSize = false"]))
    print("dist                             :", run(c, "./build-dist/katago",
                                                    ["requireMaxBoardSize = false",
                                                     "distServers = 127.0.0.1:%d" % REMOTE_DS_PORT]))
finally:
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()
