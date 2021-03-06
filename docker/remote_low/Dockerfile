FROM nvidia/cuda:9.0-cudnn7-devel

LABEL  maintainer walkingmask@gmail.com

USER root

ARG U_NAME=walkingmask
ARG U_ID=1000
ARG G_ID=1000

RUN apt-get update && \
    apt-get install -y --allow-downgrades --no-install-recommends \
        wget \
        git \
        vim \
        openssl \
        ca-certificates \
        sudo \
        bzip2 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libcudnn7=7.0.4.31-1+cuda9.0 \
        libcudnn7-dev=7.0.4.31-1+cuda9.0 \
        && \
    apt-get clean

RUN groupadd $U_NAME -g $G_ID && \
    useradd $U_NAME -m -u $U_ID -g $G_ID && \
    echo "%$U_NAME ALL=(ALL) NOPASSWD: ALL" >>/etc/sudoers.d/$U_NAME

COPY ["jupyter_notebook_config.py", "jl", "tb", "/"]
RUN mkdir /home/$U_NAME/.jupyter && \
    mv /jupyter_notebook_config.py /home/$U_NAME/.jupyter/ && \
    chown -R $U_NAME:$U_NAME /home/$U_NAME/.jupyter && \
    mv /jl /usr/local/bin/ && \
    chmod 755 /usr/local/bin/jl && \
    mv /tb /usr/local/bin/ && \
    chmod 755 /usr/local/bin/tb

USER $U_NAME
ENV HOME /home/$U_NAME
WORKDIR $HOME

RUN mkdir .pyenv && git clone git://github.com/yyuu/pyenv.git .pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/versions/anaconda3-5.0.0/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install anaconda3-5.0.0 && \
    pyenv global anaconda3-5.0.0 && \
    pyenv rehash && \
    pip install -U pip && \
    conda update -y conda && \
    pip install \
        jupyterlab \
        'imageio==2.2.0' \
        'Keras==2.1.6' \
        'matplotlib==2.0.2' \
        'numpy==1.14.3' \
        'opencv-python==3.4.1.15' \
        'pygame==1.9.4' \
        'scikit-image==0.13.0' \
        'scikit-learn==0.19.0' \
        'tensorflow-gpu==1.5.0' \
    && \
    git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/ntasfi/PyGame-Learning-Environment.git ./ple && \
    cd ple && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/lusob/gym-ple.git && \
    cd gym-ple && \
    pip install -e . && \
    cd ..

EXPOSE 8888
EXPOSE 6006
