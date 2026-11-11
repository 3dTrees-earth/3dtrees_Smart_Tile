# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# Pipeline uses laspy + PDAL + untwine. Untwine is the default COPC conversion
# strategy (fast). The untwine binary is rebuilt from the 1.5.1 release with
# the patches in docker/untwine (see docker/untwine/README.md).
# =============================================================================

FROM condaforge/miniforge3:24.11.3-2@sha256:1148ac0d3edaa4f25ca77d35e440766883a1be275fd5552f106c4756ac53f532 AS runtime

# Python 3.12, PDAL/GDAL/Untwine, and every Python dependency are installed
# from the generated explicit conda lock, including immutable artifact hashes.
COPY conda-linux-64.lock /tmp/conda-linux-64.lock
# Keep the locked runtime separate from the base package manager: the lock
# may replace libraries needed by the bootstrap mamba itself.
RUN mamba create -p /opt/smarttile --file /tmp/conda-linux-64.lock -y && \
    mamba clean --all -y && \
    rm /tmp/conda-linux-64.lock

# -----------------------------------------------------------------------------
# Patched untwine, compiled against the locked runtime's own PDAL headers and
# libraries so the binary matches the libpdal it runs with.
# -----------------------------------------------------------------------------
FROM runtime AS untwine

ARG UNTWINE_VERSION=1.5.1
ARG UNTWINE_SHA256=69b5ef26a28c514b1a8459440f32faccc4fb2d3b835aa9733bf4d8c31a5f7022
RUN mamba create -p /opt/build -c conda-forge -y cmake make patch curl "gxx_linux-64=14" && \
    mamba clean --all -y
COPY docker/untwine/*.patch /tmp/untwine-patches/
RUN cd /tmp && \
    /opt/build/bin/curl -fsSL -o untwine.tar.gz "https://github.com/hobuinc/untwine/archive/refs/tags/${UNTWINE_VERSION}.tar.gz" && \
    echo "${UNTWINE_SHA256}  untwine.tar.gz" | sha256sum -c - && \
    tar xzf untwine.tar.gz && \
    cd "untwine-${UNTWINE_VERSION}" && \
    for p in /tmp/untwine-patches/*.patch; do /opt/build/bin/patch -p1 < "$p"; done && \
    PATH="/opt/build/bin:${PATH}" cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/smarttile \
        -DCMAKE_CXX_COMPILER=/opt/build/bin/x86_64-conda-linux-gnu-g++ \
        -DCMAKE_INSTALL_RPATH=/opt/smarttile/lib && \
    PATH="/opt/build/bin:${PATH}" cmake --build build -j 8 && \
    install -m 0755 build/bin/untwine /tmp/untwine

FROM runtime

COPY --from=untwine /tmp/untwine /opt/smarttile/bin/untwine
LABEL org.3dtrees.untwine.version="1.5.1+3dtrees.1"

# Use the locked runtime for every subsequent build check and tool command.
ENV PATH="/opt/smarttile/bin:${PATH}"

# Fail the build if PDAL or the patched Untwine is missing
RUN pdal --version
RUN untwine --version | grep -F '(1.5.1+3dtrees.1)'
RUN python --version | grep '^Python 3\.12\.'

# ===========================================
# Setup project
# ===========================================
WORKDIR /src

# Copy Python scripts from src/ folder
COPY src/ /src/

# Data mount points for manual runs
RUN mkdir -p /in /out /src/out
RUN chmod -R a+rX /src && chmod -R 755 /in /out /src/out

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/src
ENV LD_LIBRARY_PATH="/opt/smarttile/lib"
# Fix PROJ database path (conda location)
ENV PROJ_DATA="/opt/smarttile/share/proj"
ENV PROJ_LIB="/opt/smarttile/share/proj"
# Fix matplotlib config directory (writable location)
ENV MPLCONFIGDIR="/tmp/matplotlib"


# Usage (no ENTRYPOINT; Galaxy runs `python /src/run.py`):
#   docker build -t smarttile .
#   docker run --rm -v "$PWD/data:/data" smarttile python /src/run.py \
#       --task tile --input-dir /data/in --output-dir /data/out --tile-outputs 0.1:laz
#   docker run --rm smarttile python /src/run.py --help
