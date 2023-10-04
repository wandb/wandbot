git pull && \
python -m pip install -U poetry && \
poetry install --all-extras --no-cache && \
poetry build && \
rm -rf $POETRY_CACHE_DIR/* && \
echo "Running applications" && \
mkdir -p ./data/cache && \
(poetry run uvicorn wandbot.api.app:app --host="0.0.0.0" --port=8000 > api.log 2>&1) & \
(poetry run python -m wandbot.apps.slack > slack_app.log 2>&1) & \
(poetry run python -m wandbot.apps.discord > discord_app.log 2>&1)