"""Open the PR in SakuraGoH/KataGo-Distributed-Inferring using the stored git credential.

The token is read from the git credential helper in memory and never printed.
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "SakuraGoH/KataGo-Distributed-Inferring"
HEAD = "docs/distributed-inference"
BASE = "main"
TITLE = "Distributed inference: remote NN servers over TCP (protocol v2)"
BODY = Path(r"C:\Users\10632\Desktop\KataGo-master\docs\PULL_REQUEST.md").read_text(encoding="utf-8")


def token():
    out = subprocess.run(["git", "credential", "fill"], input=b"protocol=https\nhost=github.com\n\n",
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode()
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no stored github credential")


def api(method, url, tok, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + tok)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "dsh")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


tok = token()
code, payload = api("GET", "https://api.github.com/repos/%s/pulls?state=open&head=SakuraGoH:%s"
                    % (REPO, HEAD), tok)
if code == 200:
    for pr in payload:
        if pr.get("head", {}).get("ref") == HEAD:
            print("PR already open:", pr["html_url"])
            sys.exit(0)

code, payload = api("POST", "https://api.github.com/repos/%s/pulls" % REPO, tok, {
    "title": TITLE, "head": HEAD, "base": BASE, "body": BODY,
})
if code in (200, 201):
    print("PR created:", payload["html_url"])
else:
    print("FAILED", code, payload.get("message"))
    print(json.dumps(payload.get("errors", []), ensure_ascii=False))
    sys.exit(1)
