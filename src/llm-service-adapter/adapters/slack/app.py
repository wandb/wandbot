import os
from slack_bolt import App
from.config import INTRO_MESSAGE, OUTRO_MESSAGE

def create_app(slack_adapter):

    app = App(token=os.environ.get("SLACK_APP_TOKEN"))

    @app.event("app_mention")
    def command_handler(body, say, logger):
        try:
            text = body['event'].get('text')
            user = body["event"].get("user")
            thread_ts = body["event"].get("thread_ts", None) or body["event"].get(
            "ts", None)
            text = " ".join(text.split()[1:])
            if thread_ts is not None:
                say(text=f"Hi <@{user}>:\n\n{INTRO_MESSAGE}",
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                    thread_ts=thread_ts)
            else:
                say(
                    text=f"Hi <@{user}>:\n\n{INTRO_MESSAGE}",
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                )
            query, response, timings = slack_adapter.process_query(text)
            logger.info(response)
            if thread_ts is not None:
                say(text=response,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                    thread_ts=thread_ts)

                say(text=OUTRO_MESSAGE,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                    thread_ts=thread_ts)
            else:
                say(
                    text=response,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                )
                say(
                    text=OUTRO_MESSAGE,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                )
        except Exception as e:
            logger.error(f"Error posting message: {e}")

    @app.event("reaction_added")
    def handle_reaction_added(event, say):
        channel_id = event["item"]["channel"]
        message_ts = event["item"]["ts"]
        result = app.client.conversations_history(
            channel=channel_id, latest=message_ts, limit=1, inclusive=True
        )

        if result["ok"] and len(result["messages"]) > 0:
            message = result["messages"][0]
            print(f"A user reacted with {event['reaction']} to message: {message['text']}")
        else:
            print(
                f"Unable to retrieve message for reaction {event['reaction']} in channel {channel_id}"
            )

    return app