import asyncio
import functools
import logging
import os
import sqlite3
import typing

import discord
import wandb
from chat import Chat
from config import default_config, TEAM, PROJECT, JOB_TYPE
from discord.ext import commands

WAIT_TIME = 300.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

intents = discord.Intents.all()
intents.typing = False
intents.presences = False
intents.messages = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)

wandb_run = wandb.init(
    entity=TEAM,
    project=PROJECT,
    job_type=JOB_TYPE,
    config=default_config,
)

chat = Chat(model_name=default_config.model_name, wandb_run=wandb_run)

# Create and connect to the SQLite database
conn = sqlite3.connect("responses.db")
cursor = conn.cursor()

# Create a table in the database for storing user questions, bot responses, and reactions
cursor.execute(
    """CREATE TABLE IF NOT EXISTS responses (
                    discord_id INTEGER,
                    wandb_run_id TEXT,
                    question TEXT,
                    response TEXT,
                    feedback TEXT
                  )"""
)


async def run_chat(blocking_func: typing.Callable, *args, **kwargs) -> typing.Any:
    """Runs a blocking function in a non-blocking way"""
    func = functools.partial(
        blocking_func, *args, **kwargs
    )  # `run_in_executor` doesn't support kwargs, `functools.partial` does
    return await bot.loop.run_in_executor(None, func)


INTRO_MESSAGE = f"""Please note that **wandbot is currently in alpha testing** and will experience frequent updates.\n\nPlease do not share any private or sensitive information in your query at this time.\n\nGenerating response... 🤖\n\n"""

OUTRO_MESSAGE = f"""```If you still need help please try re-phrase your question, or alternatively reach out to the Weights & Biases Support Team at support@wandb.com \n\n Was this response helpful? Please react below to let us know```"""


@bot.event
async def on_ready():
    logger.info(f"We have logged in as {bot.user}")
    logger.info(
        f"Servers connected: {len(bot.guilds)}"
    )  # Add this line to see the number of servers the bot is connected to


@bot.event
async def on_message(message: discord.Message):
    logger.info("Mentioned in message")
    if message.author == bot.user:
        return
    if bot.user is not None and bot.user.mentioned_in(message):
        mention = f"<@{message.author.id}>"
        thread = await message.channel.create_thread(
            name=f"Thread", type=discord.ChannelType.public_thread
        )  # currently calling it "Thread" because W&B Support makes it sound too official.
        await thread.send(f"Hi {mention}: {INTRO_MESSAGE}", mention_author=True)
        response = await run_chat(chat, message.clean_content)
        sent_message = await thread.send(response)
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
            await thread.send("")
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
        cursor.execute(
            f"INSERT INTO responses (discord_id,  wandb_run_id, question, response, feedback) VALUES (?, ?, ?, ?, ?)",
            (message.author.id, chat.wandb_run.id, message.content, response, feedback),
        )
        conn.commit()

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))
