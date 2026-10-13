#!/usr/bin/env bash
# Verify a GitHub Release (draft or public) has the required subject set.
# Draft releases are invisible to contents:read tokens; callers must supply
# a token that can see the release.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
  echo "usage: $(basename "$0") vX.Y.Z [owner/repo]" >&2
  exit 1
}
tag="$1"
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "tag must be vX.Y.Z" >&2
  exit 1
}
version="${tag#v}"
repository="${2:-${GITHUB_REPOSITORY:-chameleon-ai/openasr}}"
command -v gh >/dev/null 2>&1 || { echo "gh is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

pack_dir="$(mktemp -d)"
actual_file="$(mktemp)"
cleanup() { rm -rf -- "$pack_dir" "$actual_file"; }
trap cleanup EXIT

gh release view "$tag" --repo "$repository" --json assets \
  --jq '.assets[].name' | sort > "$actual_file"
# A release carries all 21 Windows backend packs or none (skip-windows):
# derive the expectation from the release's own asset list so a half-built
# matrix can never compare clean. Anything in between fails both branches
# (full-name download below errors on the first missing pack).
if grep -q '^backend-pack-.*\.json$' "$actual_file"; then
  pack_args=()
else
  pack_args=(--skip-windows)
fi
python3 tooling/release-manifest/gh_release.py download-packs "$tag" "$pack_dir" --repo "$repository" "${pack_args[@]}"
python3 tooling/release-manifest/release_completeness.py compare \
  --version "$version" \
  --pack-dir "$pack_dir" \
  --actual-file "$actual_file" \
  "${pack_args[@]}"
