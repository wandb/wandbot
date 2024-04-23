(uvicorn wandbot.api.app:app --host="0.0.0.0" --port=8000) & \
(python -m wandbot.apps.slack -l en) & \
(python -m wandbot.apps.slack -l ja) & \
(python -m wandbot.apps.discord) & \
