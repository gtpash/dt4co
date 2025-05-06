##################################################################################################
# 
# This Dockerfile is used to build the compute Docker image for TACC's Frontera system.
# 
# In particular, this will equip the PETSc / FEniCS build with the following tools:
#   - Boost 1.74.0
#   - SVMTK
#   - h5py
#   - nibabel
#   - meshio
#   - neatmesh
#   - dcm2niix
#   - itk-elastix (see the note below)
#   - SimpleITK
# 
# The image is available at: https://hub.docker.com/r/gtpash/onco-fenics-2019.1.0:latest
# 
##################################################################################################

FROM gtpash/tacc-ubuntu18-mvapich2.3-ib-python3.11:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y libglu1-mesa libxcursor-dev

# Upgrade CMake.
RUN apt-get update \
    && apt remove cmake -y \
    && pip3 install cmake==3.22.0

# Install all of boost.
RUN apt-get update \
    && apt-get -y --purge remove \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev

WORKDIR /home/

RUN wget -q https://archives.boost.io/release/1.74.0/source/boost_1_74_0.tar.gz \
    && tar -zxf boost_1_74_0.tar.gz \
    && cd boost_1_74_0 \
    && ./bootstrap.sh \
    && ./b2 --with=all -j 4 install \
    && cat /usr/local/include/boost/version.hpp | grep "BOOST_LIB_VERSION"

# Install SVMTK.
RUN git clone --recursive -b v2.0-beta https://github.com/SVMTK/SVMTK && \
    cd SVMTK && \
    python3 setup.py install 


# set up the python environment

# h5py needs to be built against the HDF5 that came with PETSc
ENV HDF5_DIR=${HDF5_ROOT}

# ITK-Elastix is missing lbbtb, will be fixed in 5.4.1
# https://github.com/InsightSoftwareConsortium/ITK/issues/4713

# Set up python environment for oncology project.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir pytest \
    && python3 -m pip install --no-cache-dir seaborn \
    && python3 -m pip install --no-cache-dir nibabel \
    && python3 -m pip install --no-cache-dir meshio \
    && CC="mpicc" HDF5_MPI="ON" HDF5_DIR=${HDF5_DIR} pip install --no-cache-dir --no-binary=h5py h5py \
    && python3 -m pip install --no-cache-dir neatmesh \
    && python3 -m pip install --no-cache-dir dcm2niix

RUN python3 -m pip install --no-cache-dir itk-elastix \
    && python3 -m pip install --no-cache-dir SimpleITK

