pacman -Su python python-pip
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r req.txt
python -m ipykernel install --user