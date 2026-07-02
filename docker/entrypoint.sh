#!/usr/bin/env bash
set -euo pipefail

cd /workspace/impact

usage() {
    cat <<'EOF'
IMPACT Docker commands:
  box [args...]       Run the box-pushing CITO example.
  push_t [args...]    Run the Push-T CITO example.
  cart [args...]      Run the cart-transporter CITO example.
  all                 Run all three CITO examples with default arguments.
  toy_mpcc            Run the smallest generic MPCC example.
  bash                Open an interactive shell in the image.

Default arguments:
  box      0 0 0  1 1 0
  push_t   0 0 0  0 0 1
  cart     0 0 0 0  1.5 1 0 0

Mount results out of the container with:
  docker run --rm -v "$PWD/results:/workspace/impact/results" impact box
EOF
}

run_box() {
    if [ "$#" -eq 0 ]; then
        set -- 0 0 0 1 1 0
    fi
    ./build/experiments/box/box_impact_multiple "$@"
}

run_push_t() {
    if [ "$#" -eq 0 ]; then
        set -- 0 0 0 0 0 1
    fi
    ./build/experiments/push_t/push_t_impact_multiple "$@"
}

run_cart() {
    if [ "$#" -eq 0 ]; then
        set -- 0 0 0 0 1.5 1 0 0
    fi
    ./build/experiments/cart_transporter/cart_transporter_impact_multiple "$@"
}

cmd="${1:-box}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$cmd" in
    box)
        run_box "$@"
        ;;
    push_t|pusht|push-t)
        run_push_t "$@"
        ;;
    cart|cart_transporter|cart-transporter)
        run_cart "$@"
        ;;
    all)
        run_box
        run_push_t
        run_cart
        ;;
    toy_mpcc|toy)
        ./build/experiments/toy_mpcc/toy_mpcc "$@"
        ;;
    bash|shell)
        exec /bin/bash "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        exec "$cmd" "$@"
        ;;
esac
