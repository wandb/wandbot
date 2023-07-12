import asyncio
import logging
import uuid

import discord
from discord.ext import commands
from wandbot.api.client import AsyncAPIClient
from wandbot.api.schemas import APIQueryResponse
from wandbot.apps.discord.config import DiscordAppConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

intents = discord.Intents.all()
intents.typing = False
intents.presences = False
intents.messages = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)
config = DiscordAppConfig()
api_client = AsyncAPIClient(url=config.WANDBOT_API_URL)


def format_response(response: APIQueryResponse | None, outro_message: str = "") -> str:
    if response is not None:
        result = response.answer
        if response.model != "gpt-4":
            warning_message = (
                f"**Warning: Falling back to {response.model}**, These results may nor be as good as "
                f"**gpt-4**\n\n"
            )
            result = warning_message + response.answer

        if config.include_sources and response.sources:
            sources_list = [
                item
                for item in response.sources.split(",")
                if item.strip().startswith("http")
            ]
            if len(sources_list) > 0:
                result = (
                    f"{result}\n\n*References*\n\n"
                    + ">"
                    + "\n".join(sources_list)
                    + "\n\n"
                )
        if outro_message:
            result = f"{result}\n\n{outro_message}"

    else:
        result = config.ERROR_MESSAGE
    return result


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

        chat_history = await api_client.get_chat_history(
            application=config.APPLICATION, thread_id=str(thread.id)
        )
        if not chat_history:
            await thread.send(
                f"🤖 Hi {mention}: {config.INTRO_MESSAGE}", mention_author=True
            )

        response = await api_client.query(
            question=str(message.clean_content),
            chat_history=chat_history,
        )
        if response is None:
            await thread.send(
                f"🤖 {mention}: {config.ERROR_MESSAGE}", mention_author=True
            )
            return
        sent_message = await thread.send(
            f"🤖 {format_response(response, config.OUTRO_MESSAGE)}"
        )

        await api_client.create_question_answer(
            thread_id=str(thread.id),
            question_answer_id=str(sent_message.id),
            **response.dict(),
        )
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
            rating = 0

        else:
            # Get the feedback value
            if str(reaction.emoji) == "👍":
                rating = 1
            elif str(reaction.emoji) == "👎":
                rating = -1
            else:
                rating = 0

        # Send feedback to API
        await api_client.create_feedback(
            feedback_id=str(uuid.uuid4()),
            question_answer_id=str(sent_message.id),
            rating=rating,
        )

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(config.DISCORD_BOT_TOKEN)
