FROM mambaorg/micromamba:1.5.10-jammy

# Build and run as root throughout. The base image defaults to a non-root
# user, but the legacy (non-BuildKit) builder creates WORKDIR/COPY content
# owned by root, which makes the build directory unwritable for that user.
# Staying root works on every Docker setup; the only cost is that files
# written to bind-mounted volumes are root-owned on Linux hosts.
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/*

# CasADi comes from the pip wheel rather than conda-forge: the extension links
# the libcasadi that ships *inside* that wheel, which is what keeps the
# serialized-function boundary version-consistent by construction. Conda's
# casadi would be a second, different build of the same library in one process.
RUN micromamba install -y -n base -c conda-forge \
        python=3.11 \
        c-compiler \
        cxx-compiler \
        cmake \
        eigen \
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
# Headless: the visualizer must not try to open a window.
ENV MPLBACKEND=Agg

WORKDIR /workspace/impact
COPY . /workspace/impact

# Install the Python bindings. This builds the solver through scikit-build-core,
# so it is the only build step the image needs -- the C++ driver executables are
# not used by the demo.
#
# BLAS/LAPACK come from the conda env (libblas.so/liblapack.so, backed by
# OpenBLAS), which exists on x86_64 and arm64 alike -- a hardcoded
# /usr/lib/x86_64-linux-gnu path breaks Apple Silicon builds. NATIVE_ARCH is off
# so the image runs on a different machine than it was built on.
# `rm -rf build build_py` guards against a stale host build directory leaking in
# when the context is created without .dockerignore (zip downloads etc.).
RUN rm -rf build build_py \
    && micromamba run -n base python -m pip install --no-cache-dir \
        'matplotlib>=3.5' 'pillow>=9.0' \
    && micromamba run -n base env \
        CMAKE_ARGS="-DIMPACT_NATIVE_ARCH=OFF -DBLA_VENDOR=Generic -DCMAKE_PREFIX_PATH=/opt/conda" \
        python -m pip install --no-cache-dir -v .

# Make `examples` importable without shadowing the solver we just installed.
#
# This needs a .pth file rather than PYTHONPATH. `python/` holds *both* packages,
# and the source copy of `impact` in it has no compiled extension -- the .so is
# installed into site-packages/impact/ by the wheel. PYTHONPATH is always placed
# *ahead* of site-packages, so it would shadow the working solver with the
# extension-less copy and every command would die on
# `ImportError: cannot import name '_impact_core'`. Paths from a .pth are
# appended by `site` instead, which is exactly the ordering required: the
# installed `impact` wins, and `examples` is still found.
RUN micromamba run -n base python -c "import site, pathlib; \
    pathlib.Path(site.getsitepackages()[0], 'impact_examples.pth') \
        .write_text('/workspace/impact/python\n')" \
    && micromamba run -n base python -c "import impact, examples; \
    print('solver from', impact.__file__); print('tasks:', examples.task_names())"

RUN install -m 0755 docker/entrypoint.sh /usr/local/bin/impact-run

ENTRYPOINT ["/usr/local/bin/impact-run"]
CMD ["push_t"]
