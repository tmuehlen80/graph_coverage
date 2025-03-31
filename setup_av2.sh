source ./.venv/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
export PATH=$HOME/.cargo/bin:$PATH
rustup default nightly
pip install git+https://github.com/argoverse/av2-api#egg=av2

