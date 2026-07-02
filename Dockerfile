FROM mambaorg/micromamba:1.5.10-jammy

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libopenblas-dev \
        procps \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
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
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/impact

RUN micromamba run -n base cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DIMPACT_BUILD_ALLEGRO=OFF \
        -DIMPACT_BUILD_TESTS=OFF \
        -DIMPACT_NATIVE_ARCH=OFF \
        -DBLA_VENDOR=OpenBLAS \
        -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
        -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
        -DCMAKE_PREFIX_PATH=/opt/conda \
    && micromamba run -n base cmake --build build -j "$(nproc)"

USER root
RUN install -m 0755 docker/entrypoint.sh /usr/local/bin/impact-run

USER $MAMBA_USER
ENTRYPOINT ["/usr/local/bin/impact-run"]
CMD ["box"]
