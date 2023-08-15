following
https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch

This notebook uses conda environment followed by pip installs
currently "pytorch_(N)_..." !!!

Creating conda environment...
export CONDA_ENV_VERSION=r
export CONDA_ENV_NAME="pytorch_"$CONDA_ENV_VERSION"_cv2_u_madison"
conda create -y -n $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install opencv-contrib-python==4.5.5.62
pip install -U ipykernel

when starting vscode:
ensure vscode python kernel points to the kernel (select via 'view' command pallet - select python kernel)
ensure vscode ipykernel points to the right venv (select ipykernel in upper right corner)

pip install pandas
pip install importlib-resources
pip install -q segmentation_models_pytorch
pip install -qU wandb
# pip install -q scikit-learn==1.0
pip install -q scikit-learn
pip install -q plotly
pip install -q matplotlib
pip install -q albumentations
pip install -q colorama
pip install -q nbformat




