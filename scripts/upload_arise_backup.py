#!/usr/bin/env python3
"""Upload the latest ARISE backup to OneDrive via Microsoft Graph.

Pattern mirrors the day_planner upload (PUT /me/drive/root:/{path}:/content).
Usage: python3 upload_arise_backup.py
"""
import json
import urllib.request
from pathlib import Path

TOKEN_PATH = Path.home() / ".onedrive" / "graph_token.json"
BACKUPS_DIR = Path("/arise/backups")
ONEDRIVE_ROOT = "ARISE/backups"  # folder under OneDrive root


def _token():
    d = json.loads(TOKEN_PATH.read_text())
    return d["access_token"]


def _latest_backup():
    dirs = sorted(BACKUPS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def _upload_file(token, local: Path, remote_rel: str):
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{remote_rel}:/content"
    data = local.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    })
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    token = _token()
    b = _latest_backup()
    if b is None:
        print("No backups found.")
        return
    print(f"Uploading: {b.name}")

    # Upload the SQLite DB + a manifest of the problems folder
    files = {
        "arise.db": b / "arise.db",
        "problems_manifest.json": _problems_manifest(),
    }
    for name, local in files.items():
        if not local.exists():
            print(f"  skip {name} (missing)")
            continue
        remote = f"{ONEDRIVE_ROOT}/{b.name}/{name}"
        resp = _upload_file(token, local, remote)
        print(f"  ✓ {name} -> {remote} (id {resp.get('id', '')[:12]}...)")

    print("Done.")


def _problems_manifest():
    import hashlib
    import json as _json
    import io
    manifest = {}
    base = Path("/arise/problems")
    for f in sorted(base.rglob("*")):
        if f.is_file():
            # include everything EXCEPT answer.csv (secret stays server-side)
            if f.name == "answer.csv":
                continue
            manifest[str(f.relative_to(base))] = {
                "size": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
    # also record dataset download URLs for later reference
    out = io.StringIO()
    _json.dump({"problems": manifest, "note": "answer.csv intentionally excluded from cloud backup"},
               out, indent=1)
    p = Path("/tmp/arise_problems_manifest.json")
    p.write_text(out.getvalue())
    return p


if __name__ == "__main__":
    main()
