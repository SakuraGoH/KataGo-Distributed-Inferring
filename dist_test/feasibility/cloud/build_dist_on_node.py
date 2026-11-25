"""Build the DISTRIBUTED client on the node and test it against the local distserver."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

NODE = "weste"
BUILD = ("cd /root/cpp && cmake -S . -B build-dist -DUSE_BACKEND=DISTRIBUTED "
         "-DNO_GIT_REVISION=1 > /root/dist_configure.log 2>&1 && "
         "cmake --build build-dist -j48 > /root/dist_build.log 2>&1")

c = sshutil.connect(NODE)
rc, out = sshutil.run(c, "test -x /root/cpp/build-dist/katago && echo HAVE || echo NO")
if "HAVE" not in out:
    print("building dist client...", flush=True)
    sshutil.run(c, "setsid bash -c '%s' > /dev/null 2>&1 < /dev/null &" % BUILD)
    for i in range(120):
        rc, out = sshutil.run(c, "test -x /root/cpp/build-dist/katago && echo HAVE || tail -n 1 /root/dist_build.log")
        if "HAVE" in out:
            break
        if i % 10 == 0:
            print("  ", out.strip()[:100], flush=True)
        import time
        time.sleep(5)
print(sshutil.run(c, "test -x /root/cpp/build-dist/katago && echo BIN_OK || echo BIN_MISSING")[1])
c.close()
