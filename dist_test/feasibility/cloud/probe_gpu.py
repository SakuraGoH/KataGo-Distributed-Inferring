import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

setup = r"""
cd /root/cpp
for B in 1 8 32; do
  echo "--- batch $B ---"
  ./build-cuda/katago benchmarknn -model /root/model.bin.gz -config /root/bench.cfg \
      -batch-size $B -iterations 20 -warmup 2 -json 2>/dev/null | tail -1
done
"""

for h in ("weste", "westc"):
    print("#" * 12, h)
    c = sshutil.connect(h)
    rc, out = sshutil.run(c, setup, timeout=900)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            print("  batch %-2d: %8.1f evals/s  %7.2f ms/batch  threads=%d"
                  % (d["batchSize"], d["actualWallNNEvalsPerSec"],
                     d["perThreadMedianMs"][0], d["numThreads"]))
        elif line:
            print(line)
    c.close()
