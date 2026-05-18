#!/bin/bash
# Build OpenCV 4 (core + imgproc only) from source inside the manylinux2014
# container so cibuildwheel can find the C++ headers + libs when it builds
# lavan's vendored pupil_detector_2d. Mirrors upstream pupil-detectors'
# scripts/manylinux-before-all.sh; trimmed to the two modules we actually
# link against (find_package(OpenCV REQUIRED COMPONENTS core imgproc)).
set -e

git clone --depth 1 --branch 4.6.0 https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake .. -DBUILD_LIST=core,imgproc
cmake --build .
cmake --install .
