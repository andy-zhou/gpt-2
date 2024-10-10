FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG PIXI_VERSION=v0.32.1
ARG USERNAME=vscode

ADD https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl /usr/local/bin/pixi
RUN chmod +x /usr/local/bin/pixi && pixi info

RUN apt-get update && apt-get install -y \
    tmux

RUN useradd -s /bin/bash ${USERNAME}
USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN echo 'eval "$(pixi completion -s bash)"' >> /home/${USERNAME}/.bashrc