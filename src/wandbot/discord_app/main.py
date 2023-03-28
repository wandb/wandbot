import os
import discord
from discord.ext import commands
import asyncio
# from chat import Chat
import typing
import functools
import sqlite3
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


intents = discord.Intents.all()
intents.typing = False
intents.presences = False
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
# chat = Chat()

# Create and connect to the SQLite database
conn = sqlite3.connect('responses.db')
cursor = conn.cursor()

# Create a table in the database for storing user questions, bot responses, and reactions
cursor.execute('''CREATE TABLE IF NOT EXISTS responses (
                    user_id INTEGER,
                    question TEXT,
                    response TEXT,
                    feedback TEXT
                  )''')


async def run_chat(blocking_func: typing.Callable, *args, **kwargs) -> typing.Any:
    """Runs a blocking function in a non-blocking way"""
    func = functools.partial(
        blocking_func, *args, **kwargs
    )  # `run_in_executor` doesn't support kwargs, `functools.partial` does
    return await bot.loop.run_in_executor(None, func)


INTRO_MESSAGE = f"""Please note that Wandbot is currently in alpha testing and will experience frequent updates. Please do not share any private or sensitive information in your query at this time. Generating response... 🤖\n{'-'*80}\n"""

OUTRO_MESSAGE = f"""{'-'*80}
Was this response helpful? Please react with 👍or 👎
If you still need help please try re-phrase your question, 
or alternatively reach out to the Weights & Biases Support Team
at support@wandb.com
{'-'*80}"""


@bot.event
async def on_ready():
    logger.info(f"We have logged in as {bot.user}")
    logger.info(
        f"Servers connected: {len(bot.guilds)}"
    )  # Add this line to see the number of servers the bot is connected to


@bot.event
async def on_message(message):
    logger.info('Mentioned in message')
    if message.author == bot.user:
        return
    if bot.user.mentioned_in(message):
        thread = await message.channel.create_thread(
            name="Thread", type=discord.ChannelType.public_thread
        )
        await thread.send(f"Hi @{message.author}: {INTRO_MESSAGE}", mention_author=True)
        # response = await run_chat(chat, message.clean_content)
        response = 'Hello World!'
        sent_message = await thread.send(response)
        # sent_message = await thread.send(OUTRO_MESSAGE)

        # # Add reactions for feedback
        await sent_message.add_reaction('👍')
        await sent_message.add_reaction('👎')

        # # Wait for reactions
        def check(reaction, user):
            return user == message.author and str(reaction.emoji) in ['👍', '👎']

        try:
            reaction, user = await bot.wait_for('reaction_add', timeout=5.0, check=check)

        except asyncio.TimeoutError:
            await thread.send('Sorry, you took too long to give feedback.')
            feedback = "none"

        else:
            # Get the feedback value
            feedback = str(reaction.emoji)
        logger.info(f"Feedback: {feedback}")
        cursor.execute(f"INSERT INTO responses (user_id,  question, response, feedback) VALUES (?, ?, ?, ?)",
                       (message.author.id, message.content, response, feedback))
        conn.commit()

    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))
