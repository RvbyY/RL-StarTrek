sudo apt-get install -y python3 python3-pip python3-venv
python3 -m venv ~/pytorch-env
source ~/pytorch-env/bin/activate
pip install --upgrade pip
pip install numpy
pip install torch
pip install gymnasium
pip install swig
pip install imageio
pip install matplotlib
pip install gymnasium['Box2D']
python3 src/train.py --config configs/dqn_test.yaml
