
# Reproducible environment
 - use Dockerfile to maintain reproducible environments & performance
 - separate build-dependencies from release images by [multi-stage Docker build](https://zhuanlan.zhihu.com/p/687038317)
 - use `--rm` containers for testing, all changes to the persistent environment should be done inside Dockerfile, not in container.


