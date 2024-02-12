pip install fasttext && \
poetry install --without dev --all-extras  && \
poetry build && \
mkdir -p ./data/cache
