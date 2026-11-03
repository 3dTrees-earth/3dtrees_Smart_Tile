# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# The Rust engine (rust/) is built first; the runtime image carries it beside
# the Python implementation, PDAL and Untwine. `run.py` runs every task with
# the Rust engine and keeps Python for `--engine python` and unported options.
# =============================================================================

FROM rust:1.89-slim-bookworm AS rust-engine
WORKDIR /build
COPY rust/ /build/
COPY native/ /native/
RUN cargo build --release --locked && strip target/release/smarttile

FROM condaforge/miniforge3:24.11.3-2@sha256:1148ac0d3edaa4f25ca77d35e440766883a1be275fd5552f106c4756ac53f532

# Python 3.12, PDAL/GDAL/Untwine, and every Python dependency are installed
# from the generated explicit conda lock, including immutable artifact hashes.
COPY conda-linux-64.lock /tmp/conda-linux-64.lock
# Keep the locked runtime separate from the base package manager: the lock
# may replace libraries needed by the bootstrap mamba itself.
RUN mamba create -p /opt/smarttile --file /tmp/conda-linux-64.lock -y && \
    mamba clean --all -y && \
    rm /tmp/conda-linux-64.lock

# Use the locked runtime for every subsequent build check and tool command.
ENV PATH="/opt/smarttile/bin:${PATH}"

# Fail the build if PDAL or Untwine is missing
RUN pdal --version
RUN untwine --help > /dev/null 2>&1
RUN python --version | grep '^Python 3\.12\.'

# ===========================================
# Setup project
# ===========================================
WORKDIR /src

# Copy Python scripts from src/ folder
COPY src/ /src/
# The Rust engine; it loads PROJ from /opt/smarttile/lib at run time.
COPY --from=rust-engine /build/target/release/smarttile /opt/smarttile/bin/smarttile-rs
RUN /opt/smarttile/bin/smarttile-rs --task none 2>&1 | grep -q "not ported"

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
