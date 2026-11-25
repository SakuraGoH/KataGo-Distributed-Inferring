"""Is TF32 the difference? Run both sides with NVIDIA_TF32_OVERRIDE=0."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

NODE = "weste"
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def run(c, exe, extra, env=""):
    sftp = c.open_sftp()
    with sftp.open("/root/tf.cfg", "w") as f:
        f.write("\n".join(BASE + extra) + "\n")
    with sftp.open("/root/tf.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s %s gtp -model /root/model.bin.gz "
                             "-config /root/tf.cfg < /root/tf.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'"
                          % (env, exe), timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


c = sshutil.connect(NODE)
for env in ("", "NVIDIA_TF32_OVERRIDE=0 "):
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.3; true")
    sshutil.run(c, "cd /root/cpp && %ssetsid nohup ./build-cuda/katago distserver "
                   "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch 256 "
                   "> /root/distserver.log 2>&1 < /dev/null &" % (env, REMOTE_DS_PORT))
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            break
        time.sleep(0.5)
    d = run(c, "./build-dist/katago", ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT], env)
    e = run(c, "./build-cuda/katago", [], env)
    print("env=%-26s dist: %s | direct: %s" % (env.strip() or "(default)", d, e), flush=True)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
