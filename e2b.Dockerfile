# Install dependencies and customize sandbox
FROM e2bdev/code-interpreter:python-3.12.8

# Set working directory
WORKDIR /home/user

# Install Python 3.12 and set it as default
RUN apt-get update && apt-get install -y \
    wget \
    gpg \
    libstdc++6 \
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

# Set working directory
WORKDIR /home/user/wandbot

# Set LD_LIBRARY_PATH before running build.sh
RUN export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && \
    (bash build.sh || true)

RUN uv pip install --system  .
RUN poetry install

# Copy in the vector index
COPY temp_index/* /home/user/temp_index/

RUN export INDEX_DIR=$(python -c 'from wandbot.configs.vector_store_config import VectorStoreConfig; \
index_dir = VectorStoreConfig().index_dir; \
print(index_dir, end="")') && \
    mkdir -p $INDEX_DIR && \
    cp -r /home/user/temp_index/* $INDEX_DIR/ && \
    rm -rf /home/user/temp_index
