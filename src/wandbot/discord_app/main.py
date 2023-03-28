import os
import discord
from discord.ext import commands
from src.wandbot.chat import Chat

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

chat = Chat()


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    print(
        f"Servers connected: {len(bot.guilds)}"
    )  # Add this line to see the number of servers the bot is connected to


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    print(f"Message received: {message.content}")

    if bot.user.mentioned_in(message):
        print("Bot mentioned")
        await message.channel.send("Hello, I am your Discord bot!")

    await bot.process_commands(message)


token = os.getenv("DISCORD_BOT_TOKEN")
bot.run(token)
