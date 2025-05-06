############################################################
# 
# This Dockerfile is used to build an image that contains
# the necessary tools for running KUL_VBG on TACC's Frontera.
# 
# In particular, this Dockerfile installs:
#   - Ubuntu 22.04
#   - FreeSurfer 7.4.1
#   - ANTs 2.5.3
#   - MRtrix3
#   - HD-BET
#   - FSL
#   - KUL_VBG
# 
# Please modify this for your own use case.
# NOTE: Don't forget to add your own license.txt file for FreeSurfer.
# 
############################################################

FROM gtpash/tacc-ubuntu22:latest

WORKDIR /home/

# install utils
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    # for non-interactive installs
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        vim \
        ccache \ 
        cmake \
        git \
        wget \
        bc \
        dc \
        dc \
        libopenblas-dev \
        tar \
        gzip \
        libgomp1 \
        perl \
        tcsh \
        libglu1-mesa \
        libxcursor-dev \
        libsm6 \
        bash-completion \
        man \
        unzip \
        pkg-config && \
    apt-get clean && \
    docker-clean

# install miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && conda config --system --set default_threads 4 \
    && conda install --yes --no-update-deps python=3.9 \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && docker-clean

# install FreeSurfer
RUN wget -q https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz -O fs.tar.gz

RUN tar --no-same-owner -xzf fs.tar.gz && \
    rm fs.tar.gz

# Set up FreeSurfer environment
ENV FREESURFER_HOME=/home/freesurfer/
COPY license.txt $FREESURFER_HOME/license.txt

# Manually set up FreeSurfer environment (instead of using the setup script)
ENV OS=Linux
ENV PATH="${FREESURFER_HOME}/bin:${FREESURFER_HOME}/fsfast/bin:${FREESURFER_HOME}/tktools:${FREESURFER_HOME}/mni/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
ENV FREESURFER="${FREESURFER_HOME}"
ENV SUBJECTS_DIR="${FREESURFER_HOME}/subjects"
ENV LOCAL_DIR="${FREESURFER_HOME}/local"
ENV FSFAST_HOME="${FREESURFER_HOME}/fsfast"
ENV FMRI_ANALYSIS_DIR="${FREESURFER_HOME}/fsfast"
ENV FUNCTIONALS_DIR="${FREESURFER_HOME}/sessions"

# set default fs options
ENV FS_OVERRIDE=0
ENV FIX_VERTEX_AREA=""
ENV FSF_OUTPUT_FORMAT=nii.gz

# mni env requirements
ENV MINC_BIN_DIR="${FREESURFER_HOME}/mni/bin"
ENV MINC_LIB_DIR="${FREESURFER_HOME}/mni/lib"
ENV MNI_DIR="${FREESURFER_HOME}/mni"
ENV MNI_DATAPATH="${FREESURFER_HOME}/mni/data"
ENV MNI_PERL5LIB="${FREESURFER_HOME}/mni/share/perl5"
ENV PERL5LIB="${FREESURFER_HOME}/mni/share/perl5"

# install ANTs v2.5.3
RUN wget -q https://github.com/ANTsX/ANTs/releases/download/v2.5.3/ants-2.5.3-ubuntu-22.04-X64-gcc.zip -O ants.tar.gz

RUN unzip ants.tar.gz && \
    rm ants.tar.gz

# Set up ANTs environment
ENV PATH="$PATH:/home/ants-2.5.3/bin"

# install MRtrix3
RUN conda install -c mrtrix3 mrtrix3

RUN git clone https://github.com/MIC-DKFZ/HD-BET && \
    cd HD-BET && \
    python3 -m pip install .

# install FSL
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    chmod 775 fslinstaller.py && \
    python3 ./fslinstaller.py -d /usr/local/fsl/ -q

# Make FSL happy
ENV FSLDIR=/usr/local/fsl
ENV PATH=$FSLDIR/bin:$PATH
RUN /bin/bash -c 'source /usr/local/fsl/etc/fslconf/fsl.sh'
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLOUTPUTTYPE=NIFTI_GZ
# RUN ln -s /usr/local/fsl/bin/eddy_openmp /usr/local/fsl/bin/eddy

# install KUL_VBG
RUN git clone https://github.com/KUL-Radneuron/KUL_VBG.git
ENV PATH="$PATH:/home/KUL_VBG"
