(poetry run uvicorn wandbot.api.app:app --host="0.0.0.0" --port=8000) & \
(poetry run python -m wandbot.apps.slack) & \
(poetry run python -m wandbot.apps.discord)
