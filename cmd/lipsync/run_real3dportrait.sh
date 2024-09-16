#!/bin/bash
which python
python --version

# Set PYTHONPATH
export PYTHONPATH="/models/models--yerfor--Real3DPortrait/:$PYTHONPATH"
cd /models/models--yerfor--Real3DPortrait
source /opt/lipsync_venv/bin/activate
# Run the Real3DPortrait inference command
python inference/real3d_infer.py \
  --src_img "$1" \
  --drv_aud "$2" \
  --out_name "$3" \
  --drv_pose "$4" \
  --out_mode final

