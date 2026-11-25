"""Dump the NN input buffers on both sides to find the value mismatch.

Patches (on the node only, backups kept as *.orig):
  - nneval.cpp   : dump the first row the engine feeds to the backend
  - distserver.cpp: dump the first row the server received
Rebuilds the CUDA build, then compares the two dumps.
"""
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT  # noqa: E402

NODE = "weste"
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"

DUMP_ENGINE = """  if(const char* dumpPath = getenv("KATAGO_DUMP_REQ")) {
    static int dumpCount = 0;
    if(dumpCount++ == 0) {
      std::ofstream dumpFile(dumpPath, std::ios::binary);
      size_t n1 = buf.rowSpatialBuf.size();
      size_t n2 = buf.rowGlobalBuf.size();
      dumpFile.write((const char*)&n1, sizeof(n1));
      dumpFile.write((const char*)&n2, sizeof(n2));
      dumpFile.write((const char*)buf.rowSpatialBuf.data(), n1 * sizeof(float));
      dumpFile.write((const char*)buf.rowGlobalBuf.data(), n2 * sizeof(float));
    }
  }
"""

DUMP_SERVER = """              if(const char* dumpPath = getenv("KATAGO_DUMP_REQ")) {
                static int dumpCount = 0;
                if(dumpCount++ == 0) {
                  std::ofstream dumpFile(dumpPath, std::ios::binary);
                  size_t n1 = buf->rowSpatialBuf.size();
                  size_t n2 = buf->rowGlobalBuf.size();
                  dumpFile.write((const char*)&n1, sizeof(n1));
                  dumpFile.write((const char*)&n2, sizeof(n2));
                  dumpFile.write((const char*)buf->rowSpatialBuf.data(), n1 * sizeof(float));
                  dumpFile.write((const char*)buf->rowGlobalBuf.data(), n2 * sizeof(float));
                }
              }
"""


def patch(c, path, anchor, insert, includes):
    sftp = c.open_sftp()
    backup = path + ".orig"
    try:
        sftp.stat(backup)
    except IOError:
        sshutil.run(c, "cp %s %s" % (path, backup))
    with sftp.open(path, "r") as f:
        text = f.read().decode("utf-8")
    if "KATAGO_DUMP_REQ" in text:
        print("  already patched:", path)
        sftp.close()
        return True
    if anchor not in text:
        print("  ANCHOR MISSING in", path)
        sftp.close()
        return False
    for inc in includes:
        if inc not in text:
            text = inc + "\n" + text
    text = text.replace(anchor, anchor + insert, 1)
    with sftp.open(path, "w") as f:
        f.write(text)
    sftp.close()
    print("  patched:", path)
    return True


def run(c, cmd, timeout=600):
    rc, out = sshutil.run(c, cmd, timeout=timeout)
    return rc, out


def main():
    c = sshutil.connect(NODE)

    ok1 = patch(c, "/root/cpp/neuralnet/nneval.cpp",
                "  else {\n    buf.hasRowMeta = false;\n  }\n",
                DUMP_ENGINE, ["#include <fstream>"])
    ok2 = patch(c, "/root/cpp/command/distserver.cpp",
                "              conn->readFloats(buf->rowGlobalBuf.data(), (size_t)dims.numInputGlobalChannels);\n",
                "\n" + DUMP_SERVER, ["#include <fstream>"])
    if not (ok1 and ok2):
        sys.exit("patch failed")

    print("rebuilding...", flush=True)
    rc, out = run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -3", timeout=1800)
    print(out)
    if rc != 0:
        sys.exit("build failed")

    # --- run direct engine with dump ---
    sftp = c.open_sftp()
    with sftp.open("/root/dr.cfg", "w") as f:
        f.write("\n".join([
            "logToStderr = false", "rules = tromp-taylor", "allowResignation = false",
            "resignThreshold = -0.90", "resignConsecTurns = 3", "maxVisits = 1",
            "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
            "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
        ]) + "\n")
    with sftp.open("/root/dr.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = run(c, "cd /root/cpp && rm -f /root/dump_direct.bin && "
                     "KATAGO_DUMP_REQ=/root/dump_direct.bin ./build-cuda/katago gtp "
                     "-model /root/model.bin.gz -config /root/dr.cfg < /root/dr.gtp "
                     "> /dev/null 2>&1; ls -l /root/dump_direct.bin")
    print("direct dump:", out.strip())

    # --- run distserver + dist client with dump ---
    run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.5; true")
    run(c, "cd /root/cpp && rm -f /root/dump_server.bin && "
           "KATAGO_DUMP_REQ=/root/dump_server.bin setsid nohup ./build-cuda/katago distserver "
           "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch 256 "
           "-boardsize 19 -timeout 600 > /root/distserver.log 2>&1 < /dev/null &" % REMOTE_DS_PORT)
    for _ in range(60):
        rc, out = run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            break
        time.sleep(0.5)
    sftp = c.open_sftp()
    with sftp.open("/root/dc.cfg", "w") as f:
        f.write("\n".join([
            "logToStderr = false", "rules = tromp-taylor", "allowResignation = false",
            "resignThreshold = -0.90", "resignConsecTurns = 3", "maxVisits = 1",
            "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
            "nnRandomize = false", "searchRandSeed = 12345",
            "distServers = 127.0.0.1:%d" % REMOTE_DS_PORT,
        ]) + "\n")
    sftp.close()
    rc, out = run(c, "cd /root/cpp && ./build-dist/katago gtp -model /root/model.bin.gz "
                     "-config /root/dc.cfg < /root/dr.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'; "
                     "ls -l /root/dump_server.bin")
    print("dist client:", out.strip())

    rc, out = run(c, "md5sum /root/dump_direct.bin /root/dump_server.bin; "
                     "cmp /root/dump_direct.bin /root/dump_server.bin && echo IDENTICAL || echo DIFFER")
    print(out)
    run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()


if __name__ == "__main__":
    main()
