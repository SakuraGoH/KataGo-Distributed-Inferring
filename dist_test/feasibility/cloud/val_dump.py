"""Dump the raw value logits at the backend boundary on both engine and server."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

NODE = "weste"
INC = "/root/cpp/neuralnet/cudaandrocmbackend.inc"
ANCHOR = """    if(modelVersion >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
"""
DUMP = """    if(const char* valPath = getenv("KATAGO_DUMP_VAL")) {
      static int valCount = 0;
      if(valCount++ == 0) {
        std::ofstream valFile(valPath);
        valFile.precision(9);
        valFile << output->whiteWinProb << " " << output->whiteLossProb << " "
                << output->whiteNoResultProb << " " << output->whiteScoreMean << " "
                << output->whiteLead << "\\n";
      }
    }
"""

SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]

c = sshutil.connect(NODE)
sftp = c.open_sftp()
with sftp.open(INC, "r") as f:
    inc = f.read().decode("utf-8")
if "KATAGO_DUMP_VAL" in inc:
    print("already patched")
else:
    if ANCHOR not in inc:
        sys.exit("anchor missing")
    inc = inc.replace(ANCHOR, DUMP + ANCHOR, 1)
    with sftp.open(INC, "w") as f:
        f.write(inc)
    print("patched")
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -2", timeout=1800)
    print("build:", out.strip())
    if rc != 0:
        sys.exit(1)
sftp.close()

sftp = c.open_sftp()
for name, text in (("vv.cfg", "\n".join(BASE) + "\n"),
                   ("vd.cfg", "\n".join(BASE + ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT]) + "\n"),
                   ("vv.gtp", SCRIPT)):
    with sftp.open("/root/" + name, "w") as f:
        f.write(text)
sftp.close()

rc, out = sshutil.run(c, "cd /root/cpp && rm -f /root/val_direct.txt /root/val_server.txt; "
                         "KATAGO_DUMP_VAL=/root/val_direct.txt ./build-cuda/katago gtp "
                         "-model /root/model.bin.gz -config /root/vv.cfg < /root/vv.gtp "
                         "2>&1 | grep -E '^whiteWin|^whiteLead'; echo LOGITS: $(cat /root/val_direct.txt)",
                      timeout=300)
print("direct:", out.strip())

sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
sshutil.run(c, "cd /root/cpp && KATAGO_DUMP_VAL=/root/val_server.txt setsid nohup ./build-cuda/katago "
               "distserver -model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch 256 "
               "> /root/distserver.log 2>&1 < /dev/null &" % REMOTE_DS_PORT)
for _ in range(60):
    rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
    if out.strip().isdigit() and int(out.strip()) > 0:
        break
    time.sleep(0.5)
rc, out = sshutil.run(c, "cd /root/cpp && ./build-dist/katago gtp -model /root/model.bin.gz "
                         "-config /root/vd.cfg < /root/vv.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'; "
                         "echo LOGITS: $(cat /root/val_server.txt)", timeout=300)
print("dist  :", out.strip())
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
