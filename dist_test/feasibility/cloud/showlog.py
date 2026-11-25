import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

c = sshutil.connect("weste")
rc, out = sshutil.run(c, "grep -E 'Cuda backend|connection [0-9]+ from|Distserver: model|listening' /root/distserver.log", timeout=30)
lines = out.splitlines()
print("\n".join(lines[-14:]))
c.close()
