"""Dump the actual device-bound input buffer (after symmetry/layout conversion) on both
the direct engine and the distserver, to see whether the model receives the same tensor.
"""
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

NODE = "weste"
INC = "/root/cpp/neuralnet/cudaandrocmbackend.inc"
ANCHOR = """  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  CudaHandles* cudaHandles = gpuHandle->cudaHandles.get();
  cudaStream_t stream = cudaHandles->stream;
"""
DUMP = """  if(const char* dumpPath = getenv("KATAGO_DUMP_DEV")) {
    static int dumpCount = 0;
    if(dumpCount++ == 0) {
      std::ofstream dumpFile(dumpPath, std::ios::binary);
      int cfg[4] = {gpuHandle->inputsUseNHWC ? 1 : 0, gpuHandle->usingNHWC ? 1 : 0,
                    gpuHandle->nnXLen, gpuHandle->nnYLen};
      size_t n1 = (size_t)inputBuffers->singleInputElts;
      size_t n2 = (size_t)inputBuffers->singleInputGlobalElts;
      dumpFile.write((const char*)cfg, sizeof(cfg));
      dumpFile.write((const char*)&n1, sizeof(n1));
      dumpFile.write((const char*)&n2, sizeof(n2));
      dumpFile.write((const char*)inputBuffers->userInputBuffer, n1 * sizeof(float));
      dumpFile.write((const char*)inputBuffers->userInputGlobalBuffer, n2 * sizeof(float));
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
if "KATAGO_DUMP_DEV" in inc:
    print("already patched")
else:
    if ANCHOR not in inc:
        sys.exit("anchor missing")
    if "#include <fstream>" not in inc:
        inc = "#include <fstream>\n" + inc
    inc = inc.replace(ANCHOR, DUMP + ANCHOR, 1)
    with sftp.open(INC, "w") as f:
        f.write(inc)
    print("patched", INC)
sftp.close()

rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -2", timeout=1800)
print("build:", out.strip())
if rc != 0:
    sys.exit(1)

sftp = c.open_sftp()
for name, text in (("dd.cfg", "\n".join(BASE) + "\n"),
                   ("dv.cfg", "\n".join(BASE + ["distServers = 127.0.0.1:%d" % REMOTE_DS_PORT]) + "\n"),
                   ("dd.gtp", SCRIPT)):
    with sftp.open("/root/" + name, "w") as f:
        f.write(text)
sftp.close()

rc, out = sshutil.run(c, "cd /root/cpp && rm -f /root/dev_direct.bin && "
                         "KATAGO_DUMP_DEV=/root/dev_direct.bin ./build-cuda/katago gtp "
                         "-model /root/model.bin.gz -config /root/dd.cfg < /root/dd.gtp "
                         "2>&1 | grep -E '^whiteWin'; ls -l /root/dev_direct.bin", timeout=300)
print("direct:", out.strip())

sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
sshutil.run(c, "cd /root/cpp && rm -f /root/dev_server.bin && "
               "KATAGO_DUMP_DEV=/root/dev_server.bin setsid nohup ./build-cuda/katago distserver "
               "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch 256 "
               "> /root/distserver.log 2>&1 < /dev/null &" % REMOTE_DS_PORT)
for _ in range(60):
    rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
    if out.strip().isdigit() and int(out.strip()) > 0:
        break
    time.sleep(0.5)
rc, out = sshutil.run(c, "cd /root/cpp && ./build-dist/katago gtp -model /root/model.bin.gz "
                         "-config /root/dv.cfg < /root/dd.gtp 2>&1 | grep -E '^whiteWin'; "
                         "ls -l /root/dev_server.bin", timeout=300)
print("dist  :", out.strip())
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
for n in ("dev_direct.bin", "dev_server.bin"):
    sftp = c.open_sftp()
    sftp.get("/root/" + n, str(OUT / n))
    sftp.close()


def read(p):
    d = Path(p).read_bytes()
    cfg = struct.unpack_from("<4i", d, 0)
    n1, n2 = struct.unpack_from("<QQ", d, 16)
    spatial = struct.unpack_from("<%df" % n1, d, 32)
    glob = struct.unpack_from("<%df" % n2, d, 32 + 4 * n1)
    return cfg, spatial, glob


a = read(OUT / "dev_direct.bin")
b = read(OUT / "dev_server.bin")
print("\ndirect cfg (inputsNHWC, useNHWC, x, y):", a[0])
print("server cfg (inputsNHWC, useNHWC, x, y):", b[0])
print("spatial len %d/%d  global len %d/%d" % (len(a[1]), len(b[1]), len(a[2]), len(b[2])))
diff = [(i, a[1][i], b[1][i]) for i in range(min(len(a[1]), len(b[1]))) if a[1][i] != b[1][i]]
gd = [(i, a[2][i], b[2][i]) for i in range(min(len(a[2]), len(b[2]))) if a[2][i] != b[2][i]]
print("spatial differing:", len(diff), " global differing:", len(gd))
for x in diff[:6]:
    print("   ", x)
for x in gd[:6]:
    print("   g", x)
c.close()
