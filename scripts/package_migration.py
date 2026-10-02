"""Create a byte-preserving project snapshot; never include credentials/venvs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs" / "migration-20260930"
EXCLUDED_DIRS = {".venv", ".secrets", "__pycache__", ".pytest_cache", ".git", "migration-20260930"}
EXCLUDED_NAMES = {".env", "auth.json", "id_rsa", "id_ed25519", "runpod_ed25519"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "motionskeletonforcing-20260930.zip"
    if archive.exists():
        raise SystemExit("Archive already exists; do not overwrite a sent snapshot.")
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES or path.suffix in {".pyc", ".partial"}:
            continue
        files.append((path, relative.as_posix()))
    manifest = {
        "created_date": "2026-09-30",
        "source_root": str(ROOT),
        "target": "desktop-4mm681h",
        "python": platform.python_version(),
        "packages": sorted([
            {"name": d.metadata["Name"], "version": d.version}
            for d in importlib.metadata.distributions() if d.metadata["Name"]
        ], key=lambda d: d["name"].lower()),
        "excluded_dirs": sorted(EXCLUDED_DIRS),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "excluded_suffixes": [".pyc", ".partial"],
        "files": {},
    }
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for path, relative in files:
            data = path.read_bytes()
            manifest["files"][relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            z.writestr("motionskeletonforcing/" + relative, data)
        z.writestr("motionskeletonforcing/MIGRATION_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    # Check the compressed bytes and each manifest digest, without modifying sources.
    with zipfile.ZipFile(archive) as z:
        for relative, record in manifest["files"].items():
            assert hashlib.sha256(z.read("motionskeletonforcing/" + relative)).hexdigest() == record["sha256"], relative
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (DEST / (archive.name + ".sha256")).write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
    (DEST / "MIGRATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "files": len(files), "bytes": archive.stat().st_size, "sha256": digest}))


if __name__ == "__main__":
    main()
