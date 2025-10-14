FROM rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909

WORKDIR /app/

COPY vllm-0.10-whisper-patch.patch ./
COPY xformers-0.0.32+5f0419af.d20251013-cp39-abi3-linux_x86_64.whl ./

WORKDIR /usr/local/lib/python3.12/dist-packages/vllm
RUN git apply /app/vllm-0.10-whisper-patch.patch

WORKDIR /app/
RUN pip install librosa
RUN pip install xformers-0.0.32+5f0419af.d20251013-cp39-abi3-linux_x86_64.whl

