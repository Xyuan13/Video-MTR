#!/bin/bash
set -e

echo "Starting installation process..."

echo "Installing dependencies..."
# install qwen-vl-utils from the local path
pip install -e qwen-vl-utils
# same as vagen
pip install 'mathruler'
pip install 'matplotlib'
pip install 'flask'


echo "Installing flash-attn with no build isolation..."
pip install flash-attn==2.7.4.post1 --no-build-isolation
echo "Installing video-mtr package..."
pip install -e .


echo "Installation complete!"
