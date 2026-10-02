# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# Single-stage build. Pipeline uses laspy + PDAL + untwine.
# Untwine is the default COPC conversion strategy (fast).
# =============================================================================

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

# Verify PDAL and untwine
RUN pdal --version
RUN untwine --help > /dev/null 2>&1 && echo "untwine OK" || echo "WARNING: untwine not available"
RUN python --version | grep '^Python 3\.12\.'

# ===========================================
# Setup project
# ===========================================
WORKDIR /src

# Copy Python scripts from src/ folder
COPY src/ /src/

# Create a non-root user for running the application
# Create data directories with proper permissions (owned by appuser)
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



# Set entrypoint to python run.py
# ===========================================
# Usage Examples:
# ===========================================
# Build:
#   docker build -t 3dtrees-smart-tile .
#
# Filter task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task filter --segmented-folders /data/segmented --tile-bounds-json /data/tile_bounds_tindex.json --output-dir /data/output
#
# Remap task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task remap --segmented-folders /data/segmented --original-input-dir /data/original --output-dir /data/output
#
# Show parameters:
#   docker run 3dtrees-smart-tile --show-params
#
# Interactive shell:
#   docker run -it --entrypoint /bin/bash 3dtrees-smart-tile
# ===========================================
