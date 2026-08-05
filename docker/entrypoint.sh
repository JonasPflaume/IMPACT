#!/usr/bin/env bash
# Docker entry point: solve Push-T with IMPACT's Python bindings, then render the
# trajectory it found.
set -euo pipefail

cd /workspace/impact

# Anything the container writes into a bind-mounted results/ would otherwise be
# root-owned on the host: the image builds and runs as root (see the Dockerfile
# for why it cannot simply drop to a build-time user). Rather than requiring
# `--user "$(id -u):$(id -g)"` on every invocation, take the ownership of the
# results mount point -- the directory the host user created -- as the intent,
# and hand everything written underneath it back to that owner on the way out.
RESULTS_DIR=/workspace/impact/results

restore_results_ownership() {
    local owner
    # Started with --user: the writes already landed with the right owner.
    if [ "$(id -u)" != 0 ]; then
        return 0
    fi
    if [ ! -d "$RESULTS_DIR" ]; then
        return 0
    fi
    owner=$(stat -c '%u:%g' "$RESULTS_DIR" 2>/dev/null) || return 0
    # Root-owned mount point: nothing to infer, leave it alone.
    if [ "$owner" = "0:0" ]; then
        return 0
    fi
    chown -R "$owner" "$RESULTS_DIR" 2>/dev/null || true
}
trap restore_results_ownership EXIT

usage() {
    cat <<'EOF'
IMPACT Docker commands:
  push_t [args...]    Solve Push-T and render the resulting trajectory (default).
  run <task> [args]   Run any registered task; see `run list`.
  python [args...]    Start Python with the `impact` bindings importable.
  bash                Open an interactive shell in the image.

The default run is:
  python -m examples.run push_t --start 0 0 0 --goal 0.05 0.05 1.5708 --visualize

Results land in /workspace/impact/results. Mount them out with:
  docker run --rm -v "$PWD/results:/workspace/impact/results" impact
EOF
}

# `examples` is reached through the impact_examples.pth the image writes into
# site-packages, NOT through PYTHONPATH. PYTHONPATH is placed ahead of
# site-packages, and /workspace/impact/python also holds the extension-less
# *source* copy of `impact`, so exporting it here would shadow the installed
# solver and every command below would fail to import `_impact_core`.
export MPLBACKEND="${MPLBACKEND:-Agg}"

run_push_t() {
    if [ "$#" -eq 0 ]; then
        set -- --start 0 0 0 --goal 0.05 0.05 1.5708
    fi
    micromamba run -n base python -m examples.run push_t "$@" --visualize
    echo
    echo "Rendered files are under results/push_t/."
}

cmd="${1:-push_t}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$cmd" in
    push_t|pusht|push-t)
        run_push_t "$@"
        ;;
    run)
        micromamba run -n base python -m examples.run "$@"
        ;;
    # Not `exec`: replacing this shell would discard the EXIT trap above, and an
    # interactive session is exactly where results get written by hand.
    python)
        micromamba run -n base python "$@"
        ;;
    bash|shell)
        /bin/bash "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        exec "$cmd" "$@"
        ;;
esac
