"""Merge PR #1 into main (keep docs/video), then delete the PR branch and close it."""
import json
import subprocess
import sys
import urllib.error
import urllib.request

REPO = "SakuraGoH/KataGo-Distributed-Inferring"
BRANCH = "docs/distributed-inference"


def token():
    out = subprocess.run(["git", "credential", "fill"], input=b"protocol=https\nhost=github.com\n\n",
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode()
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no token")


def api(method, url, tok, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + tok)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "dsh")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, (json.loads(body) if body.strip() else {})


tok = token()
code, res = api("PUT", "https://api.github.com/repos/%s/pulls/1/merge" % REPO, tok,
                {"merge_method": "merge", "commit_title": "Merge distributed docs and validation video"})
print("merge:", code, res.get("merged"), res.get("message", ""))

code, res = api("DELETE", "https://api.github.com/repos/%s/git/refs/heads/%s" % (REPO, BRANCH), tok)
print("delete branch:", code, res.get("message", ""))

code, pr = api("GET", "https://api.github.com/repos/%s/pulls/1" % REPO, tok)
if code == 200:
    print("PR state:", pr["state"], "merged:", pr.get("merged"))
    print("PR url:", pr["html_url"])
