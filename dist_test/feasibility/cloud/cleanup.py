"""Restore the debug-patched sources on the node and rebuild clean."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

for node in ("weste",):
    c = sshutil.connect(node)
    rc, out = sshutil.run(
        c, "cd /root/cpp && for f in neuralnet/nneval.cpp command/distserver.cpp; do "
           "if [ -f $f.orig ]; then mv $f.orig $f; fi; done; "
           "grep -c KATAGO_DUMP_REQ neuralnet/nneval.cpp command/distserver.cpp || true")
    print(node, "restored, remaining refs:", out.strip())
    rc, out = sshutil.run(c, "cd /root/cpp && cmake --build build-cuda -j48 2>&1 | tail -2",
                          timeout=1800)
    print(node, "rebuild:", out.strip()[-200:])
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; rm -f /root/dump_*.bin; true")
    c.close()
