#!/bin/bash

# Check if the root directory is passed as an argument
if [ -z "$1" ]; then
  echo "Please provide the root directory as an argument."
  exit 1
fi

ROOT_DIR=$(eval echo $1)

# Define variables
CONDA_INSTALLER=Anaconda3-2024.06-1-Linux-x86_64.sh
CONDA_DIR=/models/models--yerfor--Real3DPortrait/anaconda3
REPO_URL=https://github.com/yerfor/Real3DPortrait.git
REPO_DIR=$ROOT_DIR/.lpData/models/models--yerfor--Real3DPortrait
ENV_NAME=real3dportrait

# Step 1: Download and install Anaconda
if [ ! -f $CONDA_INSTALLER ]; then
  wget https://repo.anaconda.com/archive/$CONDA_INSTALLER
fi

# Install Anaconda silently
bash ./$CONDA_INSTALLER -b -p $CONDA_DIR

# Initialize conda
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# Create the models directory if it doesn't exist
mkdir -p $ROOT_DIR/.lpData/models

# Clone the repository containing inference.py
if [ ! -d $REPO_DIR ]; then
  git clone $REPO_URL $REPO_DIR
fi

# Navigate to the repository directory
cd $REPO_DIR

# Step 2: Create and activate conda environment in the specified directory
conda create -p $CONDA_DIR/envs/$ENV_NAME python=3.9 -y
source $CONDA_DIR/bin/activate $CONDA_DIR/envs/$ENV_NAME

# Step 3: Install dependencies
conda install -p $CONDA_DIR/envs/$ENV_NAME conda-forge::ffmpeg -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -p $CONDA_DIR/envs/$ENV_NAME pytorch3d::pytorch3d -y

# Install additional dependencies
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0
pip install -r docs/prepare_env/requirements.txt -v

# Handle potential dependency conflict
pip install -r docs/prepare_env/requirements.txt -v --use-deprecated=legacy-resolver

