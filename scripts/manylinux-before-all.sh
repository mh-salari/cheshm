#!/usr/bin/env bash
# Install OpenCV and Eigen3 inside the manylinux2014 container so
# cibuildwheel can find the C++ headers + libs at wheel-build time.
# OpenCV is limited to the modules cheshm links against
# (find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)).
set -euo pipefail

OPENCV_VERSION=4.10.0
EIGEN_VERSION=3.4.0

# imgcodecs links against libpng / libjpeg; manylinux2014 lacks the -devel
# headers.
yum install -y libpng-devel libjpeg-turbo-devel

curl -L "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" | tar -xz
cmake -S "opencv-${OPENCV_VERSION}" -B opencv-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc,imgcodecs \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_OPENCV_APPS=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python_bindings_generator=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build opencv-build -j --target install

curl -L "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" | tar -xz
cmake -S "eigen-${EIGEN_VERSION}" -B eigen-build \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --install eigen-build
