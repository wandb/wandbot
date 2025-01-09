# Install dependencies and customize sandbox
FROM e2bdev/code-interpreter:latest

# Set working directory
WORKDIR /home/user

# Install Python 3.12 and set it as default
RUN apt-get update && apt-get install -y \
    wget \
    gpg \
    libstdc++6 \
    && echo "deb http://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone into workspace directory
# Pass a new value for CACHE_BUST in the docker --build-args
# to invalidate the cache from here and trigger a fresh git pull and build from here
ARG WANDBOT_COMMIT
ARG CACHE_BUST=1  
RUN git clone https://github.com/wandb/wandbot.git /home/user/wandbot && \
    cd /home/user/wandbot && \
    git checkout $WANDBOT_COMMIT

RUN pip install uv

# Set LD_LIBRARY_PATH before running build.sh
RUN cd /home/user/wandbot && \
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && \
    (bash build.sh || true)

RUN cd /home/user/wandbot && . wandbot_venv/bin/activate && uv pip install .
RUN cd /home/user/wandbot && . wandbot_venv/bin/activate && uv pip install poetry
RUN cd /home/user/wandbot && . wandbot_venv/bin/activate && poetry install

# Copy in the vector index
COPY temp_index/* /home/user/temp_index/

RUN cd /home/user/wandbot && \
    . wandbot_venv/bin/activate && \
    export INDEX_DIR=$(python -c 'from wandbot.configs.vectorstore_config import VectorStoreConfig; \
index_dir = VectorStoreConfig().index_dir; \
print(index_dir, end="")') && \
    mkdir -p $INDEX_DIR && \
    cp -r /home/user/temp_index/* $INDEX_DIR/ && \
    rm -rf /home/user/temp_index

# Ensure we're in the wandbot directory when container starts
WORKDIR /home/user


