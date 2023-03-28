import os
import discord
from discord.ext import commands
from chat import Chat
import typing
import functools

intents = discord.Intents.all()
intents.typing = False
intents.presences = False
intents.messages = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)
chat = Chat()


async def run_chat(blocking_func: typing.Callable, *args,
                   **kwargs) -> typing.Any:
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
  print(f"We have logged in as {bot.user}")
  print(f"Servers connected: {len(bot.guilds)}"
        )  # Add this line to see the number of servers the bot is connected to


@bot.event
async def on_message(message):
  if message.author == bot.user:
    return
  if bot.user.mentioned_in(message):
    thread = await message.channel.create_thread(
      name="Thread", type=discord.ChannelType.public_thread)
    await thread.send(f"Hi @{message.author}: {INTRO_MESSAGE}",
                      mention_author=True)
    response = await run_chat(chat, message.clean_content)
    await thread.send(response)
    await thread.send(OUTRO_MESSAGE)

  await bot.process_commands(message)


@bot.event
async def on_reaction_add(reaction, user):
  if reaction.message.author == bot.user:
    print(reaction)
    # async for message in reaction.message.thread.history(limit=10,
    #                                                      oldest_first=True):
    #   print(message.clean_content)

  await bot.process_commands(reaction.message)


if __name__ == "__main__":
  bot.run(os.getenv("DISCORD_BOT_TOKEN"))
