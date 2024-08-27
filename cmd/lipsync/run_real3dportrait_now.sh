#!/bin/bash
activate_conda_env() {
  local env_path="$1"
  if [ -f "${env_path}/etc/profile.d/conda.sh" ]; then
    source "${env_path}/etc/profile.d/conda.sh"
    eval "$(${env_path}/bin/conda shell.bash hook)"
    conda activate real3dportrait
  else
    echo "Conda activation script not found at ${env_path}/etc/profile.d/conda.sh"
    exit 1
  fi
}

# activate_conda_env "/models/models--yerfor--Real3DPortrait/anaconda3"
# Activate the conda environment
source /models/models--yerfor--Real3DPortrait/anaconda3/etc/profile.d/conda.sh
eval "$(/models/models--yerfor--Real3DPortrait/anaconda3/bin/conda shell.bash hook)"
conda init
source /root/.bashrc
conda activate real3dportrait
# Debugging: Check which Python is being used - 3.9 is used by the conda env for real3dportrait
which python
python --version

# Set PYTHONPATH
export PYTHONPATH="/models/models--yerfor--Real3DPortrait:$PYTHONPATH"
cd /models/models--yerfor--Real3DPortrait

# Run the Real3DPortrait inference command
python /models/models--yerfor--Real3DPortrait/inference/real3d_infer.py
