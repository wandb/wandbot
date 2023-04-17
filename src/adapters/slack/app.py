# llm-service-adapter/adapters/slack/app.py
import os
from slack_bolt import App
from slack_sdk import WebClient
from.config import INTRO_MESSAGE, OUTRO_MESSAGE
import re
from pprint import pprint

def remove_angle_brackets_and_whitespace(text):
    # Remove content inside angle brackets and the angle brackets themselves
    text_without_angle_brackets = re.sub('<[^>]*>', '', text)

    # Remove leading and trailing whitespaces
    cleaned_text = text_without_angle_brackets.strip()

    return cleaned_text


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
                response = say(text=f"Hi <@{user}>:\n\n{INTRO_MESSAGE}",
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                    thread_ts=thread_ts)
            else:
                response = say(
                    text=f"Hi <@{user}>:\n\n{INTRO_MESSAGE}",
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                )
            query, response_text, timings = slack_adapter.process_query(text)
            logger.info(response_text)
            if thread_ts is not None:
                response = say(text=response_text,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                    thread_ts=thread_ts)

                sent_message = say(text=OUTRO_MESSAGE,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                    thread_ts=thread_ts)
            else:
                response = say(
                    text=response_text,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                )
                sent_message = say(
                    text=OUTRO_MESSAGE,
                    token=os.environ.get("SLACK_BOT_TOKEN"),
                )

            start_time, end_time, elapsed_time = timings
            #TODO: use userid here or client_msg_id
            slack_adapter.add_response_to_db(
                user,
                slack_adapter.llm_service.wandb_run.id,
                query,
                response_text,
                "none",
                elapsed_time,
                start_time,
            )
            # Add reactions for feedback
            app.client.reactions_add(
                channel=body["event"]["channel"],
                timestamp=sent_message["ts"],
                name="thumbsup",
                token=os.environ.get("SLACK_BOT_TOKEN"),
            )
            app.client.reactions_add(
                channel=body["event"]["channel"],
                timestamp=sent_message["ts"],
                name="thumbsdown",
                token=os.environ.get("SLACK_BOT_TOKEN"),
            )
        except Exception as e:
            logger.error(f"Error posting message: {e}")

    @app.event("reaction_added")
    def handle_reaction_added(event, say):
        channel_id = event["item"]["channel"]
        message_ts = event["item"]["ts"]
        result = app.client.conversations_history(
            channel=channel_id, latest=message_ts, limit=1, inclusive=True, token=os.environ.get("SLACK_BOT_TOKEN")
        )

        if result["ok"] and len(result["messages"]) > 0:
            #TODO: More robust way to handle feedback. This will base it on the first message in the thread
            #BUG: Message also returns the user info so stripping that before update
            query = result["messages"][0]["text"]
            if query is not None and isinstance(query, str):
                query = remove_angle_brackets_and_whitespace(query)
            user = event["user"]
            if str(event['reaction']) in ["+1", "-1"]:
                feedback = "positive" if event['reaction'] == "+1" else "negative"
                # What happens if a different user reacts?
                slack_adapter.update_feedback_in_db(user, query, feedback)
            else:
                print(
                    f"Unhandled reaction: {event['reaction']} in channel {channel_id}"
                )
        else:
            print(
                f"Unable to retrieve message for reaction {event['reaction']} in channel {channel_id}"
            )

    return app
