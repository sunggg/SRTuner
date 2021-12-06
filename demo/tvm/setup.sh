git clone --recursive -b tvm-srtuner https://github.com/sunggg/tvm.git tvm-srtuner
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-10
pip3 install --user numpy decorator attrs tornado psutil xgboost cloudpickle mxnet
cd tvm-srtuner
mkdir build
cd build
cp ../cmake/config.cmake .
cmake ..
make -j8
