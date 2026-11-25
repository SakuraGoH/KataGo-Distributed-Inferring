"""Compare the CUDA handle log lines of the direct engine and the distserver."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT, start_distserver  # noqa: E402

NODE = "weste"
BASE = [
    "logToStderr = true", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]

c = sshutil.connect(NODE)
sftp = c.open_sftp()
with sftp.open("/root/hl.cfg", "w") as f:
    f.write("\n".join(BASE) + "\n")
with sftp.open("/root/hl.gtp", "w") as f:
    f.write("boardsize 19\nkomi 6.5\nquit\n")
sftp.close()
rc, out = sshutil.run(c, "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
                         "-config /root/hl.cfg < /root/hl.gtp 2>&1 | grep -E 'Cuda backend'", timeout=300)
print("--- direct engine ---")
print(out)

start_distserver(NODE).close()
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.5; true")
sftp = c.open_sftp()
with sftp.open("/root/hl2.cfg", "w") as f:
    f.write("\n".join(BASE + ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT]) + "\n")
with sftp.open("/root/hl2.gtp", "w") as f:
    f.write("boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n")
sftp.close()
sshutil.run(c, "cd /root/cpp && ./build-dist/katago gtp -model /root/model.bin.gz "
               "-config /root/hl2.cfg < /root/hl2.gtp > /dev/null 2>&1; true", timeout=300)
rc, out = sshutil.run(c, "grep -E 'Cuda backend' /root/distserver.log", timeout=30)
print("--- distserver ---")
print(out)
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
