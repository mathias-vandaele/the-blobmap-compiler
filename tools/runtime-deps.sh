#!/usr/bin/env bash
# blobmap-runtime may depend on blobmap-model only, never on blobc (PLAN.md, Runtime): fails
# when blobc appears anywhere in its dependency tree, normal, build or dev.
set -euo pipefail
cd "$(dirname "$0")/.."
tree="$(cargo tree --offline -q -p blobmap-runtime -e normal,build,dev --prefix none)"
if grep -E '^blobc v' <<<"$tree" >/dev/null; then
    echo "blobmap-runtime depends on blobc:"
    cargo tree --offline -q -p blobmap-runtime -e normal,build,dev -i blobc
    exit 1
fi
echo "blobmap-runtime: no dependency on blobc"
