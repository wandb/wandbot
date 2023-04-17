# llm_service_adapter/adapters/discord/app.py
import asyncio
import functools
import logging
import os
import discord
from adapters.discord.adapter import DiscordAdapter
from discord.ext import commands
from .config import PROD_DISCORD_CHANNEL_ID, TEST_DISCORD_CHANNEL_ID, WAIT_TIME, INTRO_MESSAGE, OUTRO_MESSAGE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_bot(discord_adapter):

    intents = discord.Intents.all()
    intents.typing = False
    intents.presences = False
    intents.messages = True
    intents.reactions = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        logger.info(f"We have logged in as {bot.user}")
        # print(f"We have logged in as {bot.user}")
        logger.info(
            f"Connected to {len(bot.guilds)} Discord servers"
        )  # Add this line to see the number of servers the bot is connected to
        # print(f"Servers connected: {len(bot.guilds)}")


    @bot.event
    async def on_message(message: discord.Message):
        logger.info("Mentioned in message")
        if message.author == bot.user:
            return
        if (
            bot.user is not None
            and bot.user.mentioned_in(message)
            # and (
            #     message.channel.id == PROD_DISCORD_CHANNEL_ID
            #     or message.channel.id == TEST_DISCORD_CHANNEL_ID
            # )
        ):
            mention = f"<@{message.author.id}>"
            thread = await message.channel.create_thread(
                name=f"Thread", type=discord.ChannelType.public_thread
            )  # currently calling it "Thread" because W&B Support makes it sound too official.
            await thread.send(f"🤖 Hi {mention}: {INTRO_MESSAGE}", mention_author=True)
            query, response, timings = await discord_adapter.run_chat(discord_adapter.process_query, message.clean_content)
            print("Response generated")
            start_time, end_time, elapsed_time = timings
            sent_message = await thread.send(f"🤖 {response}")
            sent_message = await thread.send(OUTRO_MESSAGE)

            # # Add reactions for feedback
            await sent_message.add_reaction("👍")
            await sent_message.add_reaction("👎")

            # # Wait for reactions
            def check(reaction, user):
                return user == message.author and str(reaction.emoji) in ["👍", "👎"]

            try:
                reaction, user = await bot.wait_for(
                    "reaction_add", timeout=WAIT_TIME, check=check
                )

            except asyncio.TimeoutError:
                await thread.send("🤖")
                feedback = "none"

            else:
                # Get the feedback value
                if str(reaction.emoji) == "👍":
                    feedback = "positive"
                elif str(reaction.emoji) == "👎":
                    feedback = "negative"
                else:
                    feedback = "none"
            logger.info(f"Feedback: {feedback}")
            await discord_adapter.add_response_to_db(
                message.author.id,
                discord_adapter.llm_service.wandb_run.id,
                query,
                response,
                feedback,
                elapsed_time,
                start_time,
            )

        await bot.process_commands(message)
    
    return bot