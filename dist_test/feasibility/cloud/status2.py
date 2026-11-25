import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

for h in ("weste", "westc"):
    c = sshutil.connect(h)
    rc, out = sshutil.run(c, "ps -eo pid,args | grep '[k]atago distserver' || echo NONE", timeout=30)
    print("###", h, "->", out.strip())
    c.close()
