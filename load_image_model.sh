#!/usr/bin/env bash

mkdir -p image_models/
cd image_models

# This looks utterly disgusting, but I've brought this into the existence by 
# modifying the example from other repo to only use one model that we need

for MODEL in 'vgg19_finetuned_all'; do
  if [ ! -f "${MODEL}.pth" ]; then
      echo "Downloading: ${MODEL}.pth"
      wget https://github.com/fabiocarrara/visual-sentiment-analysis/releases/download/torch-models/${MODEL}.pth
  else
      echo "Skipping: ${MODEL}.pth already downloaded"
  fi
done