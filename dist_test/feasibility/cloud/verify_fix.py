"""Upload the protocol-v2 fix (and pristine debug-free sources), rebuild on the node,
then compare direct vs distributed inference for convnet and transformer models."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

NODE = "weste"
CPP = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
FILES = [
    "command/distserver.cpp",
    "neuralnet/distprotocol.h",
    "neuralnet/distprotocol.cpp",
    "neuralnet/distributedbackend.cpp",
    "neuralnet/nneval.cpp",              # restore pristine (debug dump removed)
    "neuralnet/cudaandrocmbackend.inc",  # restore pristine (debug dump removed)
]
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

c = sshutil.connect(NODE)
sftp = c.open_sftp()
for f in FILES:
    sftp.put(str(CPP / f), "/root/cpp/" + f)
    print("uploaded", f)
sftp.close()

print("building build-cuda ...", flush=True)
rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -2", timeout=2400)
print(out.strip())
if rc != 0:
    sys.exit("cuda build failed")
print("building build-dist ...", flush=True)
rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-dist -j48 2>&1 | tail -2", timeout=2400)
print(out.strip())
if rc != 0:
    sys.exit("dist build failed")


def run(exe, model, extra):
    sftp = c.open_sftp()
    with sftp.open("/root/vf.cfg", "w") as f:
        f.write("\n".join(BASE + extra) + "\n")
    with sftp.open("/root/vf.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(c, "cd /root/cpp && %s gtp -model %s -config /root/vf.cfg "
                             "< /root/vf.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'" % (exe, model),
                          timeout=300)
    return " ".join(ln.strip() for ln in out.splitlines() if ln.strip())


def start(model):
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
    sshutil.run(c, "cd /root/cpp && setsid nohup ./build-cuda/katago distserver -model %s "
                   "-listen 127.0.0.1:17731 -gpus 0 -max-batch 256 > /root/distserver.log 2>&1 "
                   "< /dev/null &" % model)
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            return
        time.sleep(0.5)
    raise RuntimeError("server did not start")


ok = True
for name, model in MODELS.items():
    direct = run("./build-cuda/katago", model, [])
    start(model)
    dist = run("./build-dist/katago", model, ["distServers = 127.0.0.1:17731"])
    same = direct == dist
    ok = ok and same
    print("\n%-18s %s\n   direct: %s\n   dist  : %s" % (name, "MATCH" if same else "MISMATCH", direct, dist),
          flush=True)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")

# NCHW client on the main model
start("/root/model.bin.gz")
nchw = run("./build-dist/katago", "/root/model.bin.gz",
           ["distServers = 127.0.0.1:17731", "inputsUseNHWC = false"])
print("\nNCHW client:", nchw)
rc, out = sshutil.run(c, "grep -E 'postprocess output scale|connection 0 from' /root/distserver.log", timeout=30)
print("server:", out.strip())
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
c.close()
print("\nALL MATCH" if ok else "\nSOME MISMATCH")
