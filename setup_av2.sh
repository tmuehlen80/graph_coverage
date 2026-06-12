#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

unset VIRTUAL_ENV
VENV_PY="$PWD/.venv/bin/python"

if ! command -v rustc >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/rustc" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup default stable

"$VENV_PY" -m pip install "git+https://github.com/argoverse/av2-api#egg=av2"
