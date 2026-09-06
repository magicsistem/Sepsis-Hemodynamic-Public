#!/usr/bin/env python
from __future__ import print_function

"""Create and verify the laptop-source sidecar used on CEDIA without .git."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

EXCLUDED_PREFIXES = ("data/", "results/", "external_artifacts/")
try:
    string_types = (basestring,)
except NameError:
    string_types = (str,)


class SourceProvenanceError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_hash(files):
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tracked_source_files(root):
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    if not isinstance(output, str):
        output = output.decode("utf-8")
    return [path for path in output.split("\0") if path and not path.startswith(EXCLUDED_PREFIXES)]


def write_sidecar(root, sidecar):
    if subprocess.call(["git", "diff", "--quiet"], cwd=root) or subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=root):
        raise SourceProvenanceError("laptop source checkout is dirty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).strip()
    if not isinstance(commit, string_types):
        commit = commit.decode("ascii")
    files = {path: sha256_file(os.path.join(root, path)) for path in tracked_source_files(root)}
    payload = {"git_commit": commit, "git_dirty": False, "files": files, "source_inventory_sha256": inventory_hash(files)}
    fd, temporary = tempfile.mkstemp(prefix=".source_provenance.", dir=root)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.rename(temporary, sidecar)
    except Exception:
        os.unlink(temporary)
        raise
    return payload


def validate_sidecar(root, sidecar):
    try:
        with open(sidecar) as handle:
            payload = json.load(handle)
    except (IOError, ValueError) as error:
        raise SourceProvenanceError("invalid source provenance sidecar: {0}".format(error))
    commit, dirty, files = payload.get("git_commit"), payload.get("git_dirty"), payload.get("files")
    if not isinstance(commit, string_types) or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SourceProvenanceError("sidecar git_commit must be a lowercase SHA-1")
    if dirty is not False or not isinstance(files, dict) or not files:
        raise SourceProvenanceError("sidecar must describe a clean nonempty source inventory")
    if payload.get("source_inventory_sha256") != inventory_hash(files):
        raise SourceProvenanceError("source inventory hash mismatch")
    for relative, expected in files.items():
        path = os.path.join(root, relative)
        if not isinstance(relative, string_types) or relative.startswith("/") or ".." in relative.split("/") or not os.path.isfile(path):
            raise SourceProvenanceError("missing source file: {0}".format(relative))
        if sha256_file(path) != expected:
            raise SourceProvenanceError("source file hash mismatch: {0}".format(relative))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", metavar="SIDECAR")
    parser.add_argument("--validate", metavar="SIDECAR")
    arguments = parser.parse_args(argv)
    if bool(arguments.write) == bool(arguments.validate):
        parser.error("choose exactly one of --write or --validate")
    root = os.getcwd()
    try:
        payload = write_sidecar(root, arguments.write) if arguments.write else validate_sidecar(root, arguments.validate)
    except SourceProvenanceError as error:
        print("FAIL: {0}".format(error), file=sys.stderr)
        return 1
    print("SOURCE_PROVENANCE_PASS commit={0} inventory={1}".format(payload["git_commit"], payload["source_inventory_sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
