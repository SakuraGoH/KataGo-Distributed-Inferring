"""Dump the postprocessing parameters on engine vs client to find the value mismatch."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

NODE = "weste"
SRC = "/root/cpp/neuralnet/nneval.cpp"
ANCHOR = """        if(
          !isfinite(probSum) ||
"""
DUMP = """        if(const char* ppPath = getenv("KATAGO_DUMP_PP")) {
          static int ppCount = 0;
          if(ppCount++ == 0) {
            std::ofstream ppFile(ppPath);
            ppFile.precision(9);
            ppFile << "mult=" << postProcessParams.outputScaleMultiplier
                   << " scoreMeanMult=" << postProcessParams.scoreMeanMultiplier
                   << " leadMult=" << postProcessParams.leadMultiplier
                   << " next=" << (int)nextPlayer
                   << " ko=" << (int)history.rules.koRule
                   << " scoring=" << (int)history.rules.scoringRule
                   << " suppressNoResult="
                   << ((history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY) ? 1 : 0)
                   << " winProb=" << winProb << " lossProb=" << lossProb
                   << " noResultProb=" << noResultProb
                   << " scoreMean=" << scoreMean << " lead=" << lead << "\\n";
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
with sftp.open(SRC, "r") as f:
    src = f.read().decode("utf-8")
if "KATAGO_DUMP_PP" not in src:
    if ANCHOR not in src:
        sys.exit("anchor missing")
    if "#include <fstream>" not in src:
        src = "#include <fstream>\n" + src
    src = src.replace(ANCHOR, DUMP + ANCHOR, 1)
    with sftp.open(SRC, "w") as f:
        f.write(src)
    print("patched nneval.cpp")
else:
    print("already patched")
sftp.close()

rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -1 && "
                         "cmake --build build-dist -j48 2>&1 | tail -1", timeout=1800)
print("build:", out.strip())
if rc != 0:
    sys.exit(1)

sftp = c.open_sftp()
for name, text in (("p1.cfg", "\n".join(BASE) + "\n"),
                   ("p2.cfg", "\n".join(BASE + ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT]) + "\n"),
                   ("p.gtp", SCRIPT)):
    with sftp.open("/root/" + name, "w") as f:
        f.write(text)
sftp.close()

rc, out = sshutil.run(c, "cd /root/cpp && rm -f /root/pp_engine.txt; "
                         "KATAGO_DUMP_PP=/root/pp_engine.txt ./build-cuda/katago gtp "
                         "-model /root/model.bin.gz -config /root/p1.cfg < /root/p.gtp "
                         "2>/dev/null | grep -E '^whiteWin|^whiteLead'; cat /root/pp_engine.txt", timeout=300)
print("engine:", out.strip())

sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
sshutil.run(c, "cd /root/cpp && setsid nohup ./build-cuda/katago distserver -model /root/model.bin.gz "
               "-listen 127.0.0.1:%d -gpus 0 -max-batch 256 > /root/distserver.log 2>&1 < /dev/null &"
               % REMOTE_DS_PORT)
for _ in range(60):
    rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
    if out.strip().isdigit() and int(out.strip()) > 0:
        break
    time.sleep(0.5)
rc, out = sshutil.run(c, "cd /root/cpp && rm -f /root/pp_client.txt; "
                         "KATAGO_DUMP_PP=/root/pp_client.txt ./build-dist/katago gtp "
                         "-model /root/model.bin.gz -config /root/p2.cfg < /root/p.gtp "
                         "2>/dev/null | grep -E '^whiteWin|^whiteLead'; cat /root/pp_client.txt", timeout=300)
print("client:", out.strip())
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
