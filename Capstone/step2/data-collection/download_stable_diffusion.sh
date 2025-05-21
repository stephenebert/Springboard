#!/usr/bin/env bash

set -euo pipefail

pip install --upgrade kaggle

if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found. Please add your Kaggle API token."
  exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

mkdir -p ../data/stable_diffusion
kaggle competitions download \
  -c stable-diffusion-image-to-prompts \
  -p ../data/stable_diffusion

unzip -o ../data/stable_diffusion/*.zip -d ../data/stable_diffusion

echo "✅ Downloaded & unpacked Stable Diffusion data into ../data/stable_diffusion/"
