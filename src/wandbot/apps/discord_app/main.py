import asyncio
import logging

import discord
from discord.ext import commands
from wandbot.api.client import AsyncAPIClient
from wandbot.api.schemas import APIFeedbackRequest, APIQueryRequest, APIQueryResponse
from wandbot.apps.discord_app.config import DiscordAppConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

intents = discord.Intents.all()
intents.typing = False
intents.presences = False
intents.messages = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)
config = DiscordAppConfig()
api_client = AsyncAPIClient()


def format_response(response: APIQueryResponse | None):
    if response is not None:
        if config.include_sources:
            result = (
                response.answer
                + "\n\n**References**\n\n"
                + "- ".join(response.sources.splitlines())
            )
        else:
            result = response.answer
    else:
        result = config.ERROR_MESSAGE
    return result


async def run_api_query(query: str, thread_id: str, event_id: str) -> APIQueryResponse:
    request = APIQueryRequest(question=query, thread_id=thread_id, event_id=event_id)
    response = await api_client.query(request)
    return response


async def send_api_feedback(feedback: str, thread_id: str, question_answer_id: str):
    request = APIFeedbackRequest(
        feedback=feedback, thread_id=thread_id, question_answer_id=question_answer_id
    )
    response = await api_client.feedback(request)
    return response


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    if (
        bot.user is not None
        and bot.user.mentioned_in(message)
        and (
            message.channel.id == config.PROD_DISCORD_CHANNEL_ID
            or message.channel.id == config.TEST_DISCORD_CHANNEL_ID
        )
    ):
        mention = f"<@{message.author.id}>"
        thread = await message.channel.create_thread(
            name=f"Thread", type=discord.ChannelType.public_thread
        )  # currently calling it "Thread" because W&B Support makes it sound too official.
        await thread.send(
            f"🤖 Hi {mention}: {config.INTRO_MESSAGE}", mention_author=True
        )
        response = await run_api_query(
            str(message.clean_content), str(thread.id), str(message.id)
        )
        await thread.send(f"🤖 {format_response(response)}")

        sent_message = await thread.send(config.OUTRO_MESSAGE)

        # # Add reactions for feedback
        await sent_message.add_reaction("👍")
        await sent_message.add_reaction("👎")

        # # Wait for reactions
        def check(reaction, user):
            return user == message.author and str(reaction.emoji) in ["👍", "👎"]

        try:
            reaction, user = await bot.wait_for(
                "reaction_add", timeout=config.WAIT_TIME, check=check
            )

        except asyncio.TimeoutError:
            await thread.send("🤖")
            feedback = "neutral"

        else:
            # Get the feedback value
            if str(reaction.emoji) == "👍":
                feedback = "positive"
            elif str(reaction.emoji) == "👎":
                feedback = "negative"
            else:
                feedback = "neutral"

        # Send feedback to API
        await send_api_feedback(feedback, str(thread.id), str(message.id))

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(config.DISCORD_BOT_TOKEN)
