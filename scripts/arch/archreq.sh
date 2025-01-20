if ! which bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash
    source ~/.bashrc 
fi

if ! which git >/dev/null 2>&1; then
    sudo pacman -S git
fi

if ! which gh >/dev/null 2>&1; then
    yay -Sy github-cli
    source ~/.bashrc 
fi

# if ! which snyk >/dev/null 2>&1; then
#     curl https://static.snyk.io/cli/latest/snyk-linux -o snyk
#     chmod +x ./snyk
#     mv ./snyk /usr/local/bin/ 
# fi

source ./venv/bin/activate 
if ! which uv >/dev/null 2>&1; then
    pip install --upgrade uv 
fi

uv pip install -r req.txt

cd frontend
bun install
cd ..

chmod +x ./backend/alpaca/apca.py
chmod +x ./run.sh
