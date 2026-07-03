FROM mambaorg/micromamba:1.5.10-jammy

# Build and run as root throughout. The base image defaults to a non-root
# user, but the legacy (non-BuildKit) builder creates WORKDIR/COPY content
# owned by root, which makes the CMake build directory unwritable for that
# user. Staying root works on every Docker setup; the only cost is that
# files written to bind-mounted volumes are root-owned on Linux hosts.
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge \
        c-compiler \
        cxx-compiler \
        cmake \
        eigen \
        casadi \
        libblas \
        liblapack \
        libopenblas \
        make \
        ninja \
        pkg-config \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib
ENV CMAKE_PREFIX_PATH=/opt/conda

WORKDIR /workspace/impact
COPY . /workspace/impact

# BLAS/LAPACK come from the conda env (libblas.so/liblapack.so, backed by
# OpenBLAS), which exists on x86_64 and arm64 alike — a hardcoded
# /usr/lib/x86_64-linux-gnu path breaks Apple Silicon builds.
# `rm -rf build` guards against a stale host build directory leaking in
# when the context is created without .dockerignore (zip downloads etc.).
RUN rm -rf build \
    && micromamba run -n base cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DIMPACT_BUILD_ALLEGRO=OFF \
        -DIMPACT_BUILD_TESTS=OFF \
        -DIMPACT_NATIVE_ARCH=OFF \
        -DBLA_VENDOR=Generic \
        -DCMAKE_PREFIX_PATH=/opt/conda \
    && micromamba run -n base cmake --build build -j "$(nproc)"

RUN install -m 0755 docker/entrypoint.sh /usr/local/bin/impact-run

ENTRYPOINT ["/usr/local/bin/impact-run"]
CMD ["box"]
