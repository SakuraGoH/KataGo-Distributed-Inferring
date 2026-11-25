"""Upload the model, sanity-check the source, then start the CUDA build on both nodes."""
import hashlib
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

LOCAL_CPP = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
MODEL = Path(r"C:\Users\10632\Desktop\model.bin.gz")
KEY_FILES = ["command/distserver.cpp", "neuralnet/distprotocol.cpp",
             "neuralnet/distprotocol.h", "command/benchmarknn.cpp", "main.cpp"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


BUILD_CMD = (
    "export PATH=/usr/local/cuda-13.0/bin:$PATH; "
    "cd /root/cpp && rm -rf build-cuda && "
    "cmake -S . -B build-cuda -DUSE_BACKEND=CUDA -DNO_GIT_REVISION=1 "
    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc "
    "-DCUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu "
    "-DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so "
    "> /root/configure.log 2>&1 && "
    "cmake --build build-cuda -j48 > /root/build.log 2>&1"
)


def one(name):
    print("=== %s ===" % name, flush=True)
    c = sshutil.connect(name)
    try:
        # 1) model: upload if missing or wrong size
        rc, out = sshutil.run(c, "stat -c '%s' /root/model.bin.gz 2>/dev/null || echo 0")
        remote_size = int(out.strip() or 0)
        if remote_size != MODEL.stat().st_size:
            t0 = time.time()
            sshutil.put(c, MODEL, "/root/model.bin.gz")
            print("uploaded model in %.1fs" % (time.time() - t0), flush=True)
        else:
            print("model already present", flush=True)

        # 2) source version check
        rc, out = sshutil.run(
            c, "cd /root/cpp && sha256sum " + " ".join(KEY_FILES))
        remote_hashes = {}
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) == 2:
                remote_hashes[parts[1]] = parts[0]
        for f in KEY_FILES:
            local = sha(LOCAL_CPP / f)
            mark = "OK" if remote_hashes.get(f) == local else "DIFF"
            print("  %-28s %s" % (f, mark), flush=True)

        # 3) start build in background
        sshutil.run(c, "pkill -f 'cmake --build' 2>/dev/null; pkill -f ccb1d 2>/dev/null; true")
        sshutil.run(c, "setsid bash -c \"%s\" > /dev/null 2>&1 < /dev/null &" % BUILD_CMD.replace('"', '\\"'))
        print("build started", flush=True)
    finally:
        c.close()


threads = [threading.Thread(target=one, args=(n,)) for n in ("weste", "westc")]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("done")
