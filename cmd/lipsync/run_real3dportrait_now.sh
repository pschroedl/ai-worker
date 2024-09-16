#!/bin/bash
which python
python --version

# Set PYTHONPATH
export PYTHONPATH="/models/models--yerfor--Real3DPortrait:$PYTHONPATH"
cd /models/models--yerfor--Real3DPortrait

source /opt/lipsync_venv/bin/activate

# Run the Real3DPortrait inference command
python /models/models--yerfor--Real3DPortrait/inference/real3d_infer.py
