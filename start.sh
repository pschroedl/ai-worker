#!/bin/bash
# Set environment variables
export DEEPCACHE=True
export PIPELINE=lipsync
export MODEL_ID=camenduru/Wav2Lip
export HUGGINGFACE_TOKEN=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

uvicorn app.main:app --log-config app/cfg/uvicorn_logging_config.json --host 0.0.0.0 --port 8000
