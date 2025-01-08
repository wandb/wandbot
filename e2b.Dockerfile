# Install dependencies and customize sandbox
FROM e2bdev/code-interpreter:latest

# Set working directory
WORKDIR /workspace

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
RUN git clone https://github.com/wandb/wandbot.git /workspace/wandbot && \
    cd /workspace/wandbot && \
    git checkout make_wandbot_great_again

RUN pip install uv

# Set LD_LIBRARY_PATH before running build.sh
RUN cd /workspace/wandbot && \
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && \
    (bash build.sh || true)

RUN cd /workspace/wandbot && . wandbot_venv/bin/activate && uv pip install .
RUN cd /workspace/wandbot && . wandbot_venv/bin/activate && uv pip install poetry
RUN cd /workspace/wandbot && . wandbot_venv/bin/activate && poetry install

# Download vector index
# Declare the build arg first & set as env var
# ARG WANDB_API_KEY
# ENV WANDB_API_KEY=$WANDB_API_KEY

# COPY download_vectordb_index.py /workspace/download_vectordb_index.py
# RUN cd /workspace/wandbot && \
#     . wandbot_venv/bin/activate && \
#     python /workspace/download_vectordb_index.py

# Copy the index files to wandbot index_dir as defined in the vectorstore config
RUN cd /workspace/wandbot && \
    . wandbot_venv/bin/activate && \
    export INDEX_DIR=$(python -c "from wandbot.configs.vectorstore_config import VectorStoreConfig;\
    index_dir = VectorStoreConfig().index_dir;\
    print(index_dir, end='')") && \
    mkdir -p $INDEX_DIR && \
    cp -r temp_index/* $INDEX_DIR/

# Ensure we're in the wandbot directory when container starts
WORKDIR /workspace/wandbot


