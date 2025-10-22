FROM rocm/dev-ubuntu-22.04:6.4.1-complete

# upgrade Python version
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive
# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less libgfortran5 \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN apt update
RUN apt -y install git

RUN pip install --upgrade pip

RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
# Successfully installed MarkupSafe-2.1.5 filelock-3.19.1 fsspec-2025.9.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.3 pytorch-triton-rocm-3.4.0 sympy-1.14.0 torch-2.8.0+rocm6.4 typing-extensions-4.15.0

RUN pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4
# Successfully installed numpy-2.1.2 xformers-0.0.32.post2


RUN pip install "cmake>=3.26.1,<4"

# for test accuracy
RUN pip install librosa datasets==3.6.0 evaluate torchcodec whisper_normalizer jiwer
RUN apt update && apt -y install ffmpeg

# manually multistage build :
#   build custom vllm inside docker container based on above image and save the whl into build-context
#   so in current base, only runtime-dependency will be installed.
# docker run -it --rm --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v ${pwd}:/build mywhisper bash /build/build-vllm.sh /build/

RUN mkdir /install
WORKDIR /install
COPY vllm-0.10.2.dev3+ga9c179c2c.rocm641-cp312-cp312-linux_x86_64.whl ./
RUN pip install ./vllm-0.10.2.dev3+ga9c179c2c.rocm641-cp312-cp312-linux_x86_64.whl

# amdsmi 
RUN cd /opt/rocm-6.4.1/share/amd_smi/ && pip install .