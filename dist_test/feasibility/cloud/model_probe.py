"""Same direct-vs-dist comparison with a standard KataGo test model."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT, start_distserver  # noqa: E402

NODE = "weste"
MODELS = [
    "/root/cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz",
    "/root/cpp/tests/models/b7c96h3tfrs-test5-cnorm.bin.gz",
]
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def run(c, exe, model, extra):
    sftp = c.open_sftp()
    with sftp.open("/root/md.cfg", "w") as f:
        f.write("\n".join(BASE + extra) + "\n")
    with sftp.open("/root/md.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s gtp -model %s -config /root/md.cfg "
                             "< /root/md.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % (exe, model),
                          timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


c = sshutil.connect(NODE)
for model in MODELS:
    start_distserver(NODE, model=model).close()
    d = run(c, "./build-dist/katago", model, ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT])
    e = run(c, "./build-cuda/katago", model, [])
    print("%s\n   dist: %s\n   direct: %s" % (Path(model).name, d, e), flush=True)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
