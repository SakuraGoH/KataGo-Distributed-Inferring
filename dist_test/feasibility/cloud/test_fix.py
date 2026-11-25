"""Upload the fixed distserver, rebuild on the node, and compare direct vs dist again."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

NODE = "weste"
LOCAL = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]
MODELS = {
    "convnet b12c192": "/root/model.bin.gz",
    "convnet b10c128": "/root/cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz",
    "transformer": "/root/cpp/tests/models/b7c96h3tfrs-test5-cnorm.bin.gz",
}


def run(c, exe, model, extra):
    sftp = c.open_sftp()
    with sftp.open("/root/fx.cfg", "w") as f:
        f.write("\n".join(BASE + extra) + "\n")
    with sftp.open("/root/fx.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s gtp -model %s -config /root/fx.cfg "
                             "< /root/fx.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % (exe, model),
                          timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


def start(c, model, env=""):
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
    sshutil.run(c, "cd /root/cpp && %ssetsid nohup ./build-cuda/katago distserver -model %s "
                   "-listen 127.0.0.1:17731 -gpus 0 -max-batch 256 "
                   "> /root/distserver.log 2>&1 < /dev/null &" % (env, model))
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            return
        time.sleep(0.5)
    raise RuntimeError("server did not start")


c = sshutil.connect(NODE)
sftp = c.open_sftp()
sftp.put(str(LOCAL / "command" / "distserver.cpp"), "/root/cpp/command/distserver.cpp")
sftp.close()
print("uploaded distserver.cpp")
rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -2", timeout=1800)
print("build:", out.strip())
if rc != 0:
    sys.exit(1)

for name, model in MODELS.items():
    direct = run(c, "./build-cuda/katago", model, [])
    start(c, model)
    dist = run(c, "./build-dist/katago", model, ["distServers = 127.0.0.1:17731"])
    print("\n%-18s\n   direct: %s\n   dist  : %s" % (name, direct, dist), flush=True)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")

# NCHW client (server now transposes) on the main model
start(c, "/root/model.bin.gz")
print("\nNCHW client:\n   dist  :", run(c, "./build-dist/katago", "/root/model.bin.gz",
                                         ["distServers = 127.0.0.1:17731", "inputsUseNHWC = false"]))
rc, out = sshutil.run(c, "grep -E 'connection 0 from' /root/distserver.log", timeout=30)
print("   server:", out.strip())
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
