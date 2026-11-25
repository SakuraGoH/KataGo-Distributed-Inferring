"""Check the other node and a couple of server flags."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def run_client(c, extra):
    sftp = c.open_sftp()
    with sftp.open("/root/np.cfg", "w") as f:
        f.write("\n".join(BASE + extra) + "\n")
    with sftp.open("/root/np.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && ./build-dist/katago gtp -model /root/model.bin.gz "
                             "-config /root/np.cfg < /root/np.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'",
                          timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


def run_direct(c):
    sftp = c.open_sftp()
    with sftp.open("/root/np2.cfg", "w") as f:
        f.write("\n".join(BASE) + "\n")
    with sftp.open("/root/np2.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
                             "-config /root/np2.cfg < /root/np2.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'",
                          timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


def start(c, maxbatch=256, extra_env=""):
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.3; true")
    sshutil.run(c, "cd /root/cpp && %ssetsid nohup ./build-cuda/katago distserver "
                   "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch %d "
                   "> /root/distserver.log 2>&1 < /dev/null &" % (extra_env, REMOTE_DS_PORT, maxbatch))
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            return
        time.sleep(0.5)


for node in ("westc", "weste"):
    c = sshutil.connect(node)
    print("###", node)
    print("  direct          :", run_direct(c))
    for mb in (256, 1):
        start(c, maxbatch=mb)
        print("  dist -max-batch %-3d:" % mb, run_client(c, ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT]))
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()
