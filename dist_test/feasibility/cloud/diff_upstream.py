import hashlib
import sys
from pathlib import Path

LOCAL = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
UP = Path(r"C:\Users\10632\Desktop\KataGo-upstream\cpp")
SKIP_DIRS = {"build-cuda", "build-dist", "build", "external"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


local_files = {}
for p in LOCAL.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(LOCAL)
    if any(part in SKIP_DIRS for part in rel.parts):
        continue
    if rel.parts[:2] == ("tests", "models"):
        continue
    local_files[rel] = p

added, modified, same = [], [], 0
for rel, p in local_files.items():
    up = UP / rel
    if not up.exists():
        added.append(rel)
    elif sha(p) != sha(up):
        modified.append(rel)
    else:
        same += 1

up_files = {p.relative_to(UP) for p in UP.rglob("*") if p.is_file()}
missing = [r for r in up_files if r not in local_files and not any(
    part in SKIP_DIRS for part in r.parts) and r.parts[:2] != ("tests", "models")]

print("same:", same, "added:", len(added), "modified:", len(modified), "missing-in-local:", len(missing))
print("\nADDED:")
for r in sorted(added):
    print("  ", r)
print("\nMODIFIED:")
for r in sorted(modified):
    print("  ", r)
print("\nMISSING:")
for r in sorted(missing):
    print("  ", r)
