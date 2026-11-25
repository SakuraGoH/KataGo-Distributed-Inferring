"""Run the patched binaries and compare the engine-side vs server-side input buffers."""
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

NODE = "weste"
OUT = Path(__file__).parent / "out"
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"

BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def cfg(dist=False):
    lines = list(BASE)
    if dist:
        lines.append("distServers = 127.0.0.1:%d" % REMOTE_DS_PORT)
    return "\n".join(lines) + "\n"


def read_dump(path):
    data = Path(path).read_bytes()
    n1, n2 = struct.unpack_from("<QQ", data, 0)
    off = 16
    spatial = struct.unpack_from("<%df" % n1, data, off)
    off += 4 * n1
    glob = struct.unpack_from("<%df" % n2, data, off)
    return spatial, glob


c = sshutil.connect(NODE)
sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.5; true")
try:
    sftp = c.open_sftp()
    for name, text in (("dr.cfg", cfg(False)), ("dc.cfg", cfg(True)), ("dr.gtp", SCRIPT)):
        with sftp.open("/root/" + name, "w") as f:
            f.write(text)
    sftp.close()

    rc, out = sshutil.run(c, "cd /root/cpp && rm -f /root/dump_direct.bin && "
                             "KATAGO_DUMP_REQ=/root/dump_direct.bin ./build-cuda/katago gtp "
                             "-model /root/model.bin.gz -config /root/dr.cfg < /root/dr.gtp "
                             "2>&1 | grep -E '^whiteWin|^whiteLead|Uncaught'; "
                             "ls -l /root/dump_direct.bin", timeout=300)
    print("direct:", out.strip())

    sshutil.run(c, "cd /root/cpp && rm -f /root/dump_server.bin && "
                   "KATAGO_DUMP_REQ=/root/dump_server.bin setsid nohup ./build-cuda/katago distserver "
                   "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch 256 "
                   "-boardsize 19 -timeout 600 > /root/distserver.log 2>&1 < /dev/null &" % REMOTE_DS_PORT)
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            break
        time.sleep(0.5)

    rc, out = sshutil.run(c, "cd /root/cpp && ./build-dist/katago gtp -model /root/model.bin.gz "
                             "-config /root/dc.cfg < /root/dr.gtp 2>&1 | grep -E '^whiteWin|^whiteLead|Uncaught'; "
                             "ls -l /root/dump_server.bin", timeout=300)
    print("dist:", out.strip())

    OUT.mkdir(exist_ok=True)
    sftp = c.open_sftp()
    for name in ("dump_direct.bin", "dump_server.bin"):
        sftp.get("/root/" + name, str(OUT / name))
    sftp.close()
finally:
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()

a_s, a_g = read_dump(OUT / "dump_direct.bin")
b_s, b_g = read_dump(OUT / "dump_server.bin")
print("spatial len: engine=%d server=%d" % (len(a_s), len(b_s)))
print("global  len: engine=%d server=%d" % (len(a_g), len(b_g)))
if len(a_s) == len(b_s):
    diffs = [(i, a_s[i], b_s[i]) for i in range(len(a_s)) if a_s[i] != b_s[i]]
    print("spatial differing elements: %d / %d" % (len(diffs), len(a_s)))
    for i, x, y in diffs[:8]:
        print("   [%d] engine=%r server=%r" % (i, x, y))
if len(a_g) == len(b_g):
    gd = [(i, a_g[i], b_g[i]) for i in range(len(a_g)) if a_g[i] != b_g[i]]
    print("global differing elements: %d / %d" % (len(gd), len(a_g)))
    for i, x, y in gd[:12]:
        print("   g[%d] engine=%r server=%r" % (i, x, y))
