pip install fasttext && \
poetry install --all-extras  && \
pip install protobuf==3.19.6 && \
poetry build && \
mkdir -p ./data/cache