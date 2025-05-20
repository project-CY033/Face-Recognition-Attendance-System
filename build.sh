#!/usr/bin/env bash
# Exit on error
set -o errexit

# Create required directories
mkdir -p uploads

# Create OpenCV haarcascades directory and download required files
mkdir -p haarcascades
cd haarcascades
wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
cd ..

echo "Build completed successfully"