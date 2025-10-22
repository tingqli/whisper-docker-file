#!/bin/bash

# can be run inside docker build environment as a manually multi-stage build
# when this script is running inside a docker container, all build-dependencies installed
# during build-process will lost after container exit, only the last wheel package will be left

# docker run -it --rm --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v ${pwd}:/build mywhisper bash /build/build-vllm.sh /build/

VLLM_REPO=https://github.com/FionaZZ92/vllm.git
VLLM_BRANCH=v0.10.1_xformers
DST_PATH=$1

export PYTORCH_ROCM_ARCH="gfx942"
git clone -b $VLLM_BRANCH $VLLM_REPO
cd vllm
pip install -r ./requirements/rocm.txt
python3 setup.py bdist_wheel --dist-dir=dist
cp ./dist/*.whl $DST_PATH
