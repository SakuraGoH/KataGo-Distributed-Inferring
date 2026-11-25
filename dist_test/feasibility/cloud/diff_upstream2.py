import hashlib
from pathlib import Path

LOCAL = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
UP = Path(r"C:\Users\10632\Desktop\KataGo-upstream\cpp")
SKIP_DIRS = {"build-cuda", "build-dist", "build", "external"}


def norm_sha(p):
    data = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


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
    elif norm_sha(p) != norm_sha(up):
        modified.append(rel)
    else:
        same += 1

print("same:", same, "added:", len(added), "modified:", len(modified))
print("ADDED:")
for r in sorted(added):
    print("  ", r)
print("MODIFIED:")
for r in sorted(modified):
    print("  ", r)
