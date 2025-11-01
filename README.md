
# Reproducible environment
 - use Dockerfile to maintain reproducible environments & performance
 - separate build-dependencies from release images by [multi-stage Docker build](https://zhuanlan.zhihu.com/p/687038317)
 - use `--rm` containers for testing, all changes to the persistent environment should be done inside Dockerfile, not in container.


# audio_length_adjuster.py

a Python tool generated using deepseek.

```bash
# adjust audio file length to 600 seconds by truncating or repeating
audio_length_adjuster.py input.mp3 output.mp3 600
```
# test_fast_whisper.py

```bash
pip install viztracer
# use viztracer to generate python profiling json log
viztracer test_fast_whisper.py

# use Nsight System to generate CUDA API & kernel tracing log
/opt/nvidia/nsight-systems/2025.5.1/bin/nsys profile --trace=cuda python3 test_fast_whisper.py
```