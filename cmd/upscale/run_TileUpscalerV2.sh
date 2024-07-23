#!/bin/bash

# activate env with repo python deps
source /models/TileUpscalerV2/venv/bin/activate
# Debugging: Check which Python is being used - 3.9 is used by the conda env for real3dportrait
which python
python --version

# Set PYTHONPATH
export PYTHONPATH="/models/TileUpscalerV2:$PYTHONPATH"
cd /models/TileUpscalerV2


INPUT_IMAGE="$1"
OUTPUT_IMAGE="$2"
RESOLUTION="$3"
NUM_INFERENCE_STEPS="$4"
STRENGTH="$5"
HDR="$6"
GUIDANCE_SCALE="$7"
CONTROLNET_STRENGTH="$8"
SCHEDULER_NAME="$9"

# Run the Python script with the parameters
python main.py $INPUT_IMAGE $OUTPUT_IMAGE --resolution $RESOLUTION --num_inference_steps $NUM_INFERENCE_STEPS --strength $STRENGTH --hdr $HDR --guidance_scale $GUIDANCE_SCALE --controlnet_strength $CONTROLNET_STRENGTH --scheduler_name $SCHEDULER_NAME