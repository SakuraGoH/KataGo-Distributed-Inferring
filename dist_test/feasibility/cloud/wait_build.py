"""Wait until the CUDA build finishes on both nodes; print progress."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

deadline = time.time() + 2400
done = set()
while time.time() < deadline:
    for h in ("weste", "westc"):
        if h in done:
            continue
        try:
            c = sshutil.connect(h, timeout=20)
            rc, out = sshutil.run(
                c, "test -x /root/cpp/build-cuda/katago && echo BIN_OK || echo BUILDING; "
                   "tail -n 1 /root/build.log 2>/dev/null", timeout=30)
            c.close()
            line = out.strip().splitlines()
            status = line[0] if line else "?"
            tail = line[1] if len(line) > 1 else ""
            print("%s %-8s %s" % (time.strftime("%H:%M:%S"), h, status + " " + tail[-90:]),
                  flush=True)
            if "BIN_OK" in out:
                done.add(h)
        except Exception as e:
            print("%s %-8s ssh error: %s" % (time.strftime("%H:%M:%S"), h, e), flush=True)
    if len(done) == 2:
        print("both builds finished", flush=True)
        sys.exit(0)
    time.sleep(30)
print("timeout; done=%s" % sorted(done), flush=True)
sys.exit(1)
