(uvicorn wandbot.api.app:app --host="0.0.0.0" --port=8080) & \
(python -m wandbot.apps.slack -l en) & \
(python -m wandbot.apps.slack -l ja) & \
(python -m wandbot.apps.discord)
(python -m wandbot.apps.zendesk)