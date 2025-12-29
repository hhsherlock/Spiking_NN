#!/usr/bin/env bash

source /opt/miniforge/etc/profile.d/conda.sh
conda activate spike
export CUDA_VISIBLE_DEVICES=1
python3 -u /home/yaning/Documents/Spiking_NN/mnst_learn/pipeline_validation.py

STATUS=$?

# WEBHOOK_URL=
curl -s -X POST -H "Content-type: application/json" --data '{"text": "My pretty lady my job is done '$STATUS'"}' $WEBHOOK_URL