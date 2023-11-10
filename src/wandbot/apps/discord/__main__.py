"""Discord bot for handling user queries and interacting with an API.

This module contains the main functionality for a Discord bot that listens to user messages,
detects the language of the message, creates threads for user queries, interacts with an API to get responses,
formats the responses, and sends them back to the user. It also handles user feedback on the bot's responses.

"""
import asyncio
import logging
import uuid

import discord
import langdetect
from discord.ext import commands

from wandbot.api.client import AsyncAPIClient
from wandbot.apps.discord.config import DiscordAppConfig
from wandbot.apps.utils import format_response

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


@bot.event
async def on_message(message: discord.Message):
    """Handles the on_message event in Discord.

    Args:
        message: The message object received.

    Returns:
        None
    """
    if message.author == bot.user:
        return
    if bot.user is not None and bot.user.mentioned_in(message):
        lang_code = langdetect.detect(message.clean_content)
        mention = f"<@{message.author.id}>"
        thread = None
        is_following = None
        if isinstance(message.channel, discord.Thread):
            if (
                message.channel.parent.id == config.PROD_DISCORD_CHANNEL_ID
                or message.channel.parent.id == config.TEST_DISCORD_CHANNEL_ID
            ):
                thread = message.channel
                is_following = True
        else:
            if (
                message.channel.id == config.PROD_DISCORD_CHANNEL_ID
                or message.channel.id == config.TEST_DISCORD_CHANNEL_ID
            ):
                thread = await message.channel.create_thread(
                    name=f"Thread", type=discord.ChannelType.public_thread
                )  # currently calling it "Thread" because W&B Support makes it sound too official.
                is_following = False
        if thread is not None:
            if is_following:
                chat_history = await api_client.get_chat_history(
                    application=config.APPLICATION, thread_id=str(thread.id)
                )
            else:
                chat_history = None
            if not chat_history:
                if lang_code == "ja":
                    await thread.send(
                        f"🤖 {mention}: {config.JA_INTRO_MESSAGE}",
                        mention_author=True,
                    )
                else:
                    await thread.send(
                        f"🤖 Hi {mention}: {config.EN_INTRO_MESSAGE}",
                        mention_author=True,
                    )

            response = await api_client.query(
                question=str(message.clean_content),
                chat_history=chat_history,
            )
            if response is None:
                if lang_code == "ja":
                    await thread.send(
                        f"🤖 {mention}: {config.JA_ERROR_MESSAGE}",
                        mention_author=True,
                    )
                else:
                    await thread.send(
                        f"🤖 {mention}: {config.EN_ERROR_MESSAGE}",
                        mention_author=True,
                    )
                return
            if lang_code == "ja":
                outro_message = config.JA_OUTRO_MESSAGE
            else:
                outro_message = config.EN_OUTRO_MESSAGE
            sent_message = None
            if len(response.answer) > 2000:
                answer_chunks = []
                for i in range(0, len(response.answer), 1900):
                    answer_chunks.append(response.answer[i : i + 1900])
                for i, answer_chunk in enumerate(answer_chunks):
                    response_copy = response.model_copy()
                    response_copy.answer = answer_chunk
                    if i == len(answer_chunks) - 1:
                        sent_message = await thread.send(
                            format_response(
                                config,
                                response_copy,
                                outro_message,
                                lang_code,
                            ),
                        )
                    else:
                        sent_message = await thread.send(
                            format_response(
                                config,
                                response_copy,
                                "",
                                lang_code,
                                is_last=False,
                            ),
                        )
            else:
                sent_message = await thread.send(
                    format_response(
                        config,
                        response,
                        outro_message,
                        lang_code,
                    ),
                )
            if sent_message is not None:
                await api_client.create_question_answer(
                    thread_id=str(thread.id),
                    question_answer_id=str(sent_message.id),
                    **response.model_dump(),
                )
                # # Add reactions for feedback
                await sent_message.add_reaction("👍")
                await sent_message.add_reaction("👎")

            # # Wait for reactions
            def check(reaction, user):
                return user == message.author and str(reaction.emoji) in [
                    "👍",
                    "👎",
                ]

            try:
                reaction, user = await bot.wait_for(
                    "reaction_add", timeout=config.WAIT_TIME, check=check
                )

            except asyncio.TimeoutError:
                # await thread.send("🤖")
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
