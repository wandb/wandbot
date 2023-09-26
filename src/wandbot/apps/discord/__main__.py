import asyncio
import discord
import langdetect
import logging
import uuid
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


def format_response(response: APIQueryResponse | None, outro_message: str = "", lang: str = "en") -> str:
    if response is not None:
        result = response.answer
        if "gpt-4" not in response.model:
            if lang == "ja":
                warning_message = f"*警告: {response.model}* にフォールバックします。これらの結果は *gpt-4* ほど良くない可能性があります*"
            else:
                warning_message = (
                    f"*Warning: Falling back to {response.model}*, These results may nor be as good as " f"*gpt-4*\n\n"
                )
            result = warning_message + response.answer

        if config.include_sources and response.sources:
            sources_list = [item for item in response.sources.split(",") if item.strip().startswith("http")]
            if len(sources_list) > 0:
                items = min(len(sources_list), 3)
                if lang == "ja":
                    result = f"{result}\n\n*参考文献*\n\n>" + "\n> ".join(sources_list[:items]) + "\n\n"
                else:
                    result = f"{result}\n\n*References*\n\n>" + "\n> ".join(sources_list[:items]) + "\n\n"
        if outro_message:
            result = f"{result}\n\n{outro_message}"

    else:
        result = config.ERROR_MESSAGE
    return result


@bot.event
async def on_message(message: discord.Message):
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
                    await thread.send(f"🤖 {mention}: {config.JA_INTRO_MESSAGE}", mention_author=True)
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
                    await thread.send(f"🤖 {mention}: {config.JA_ERROR_MESSAGE}", mention_author=True)
                else:
                    await thread.send(f"🤖 {mention}: {config.EN_ERROR_MESSAGE}", mention_author=True)
                return
            if lang_code == "ja":
                outro_message = config.JA_OUTRO_MESSAGE
            else:
                outro_message = config.EN_OUTRO_MESSAGE
            sent_message = None
            if len(response.answer) > 2000:
                for i in range(0, len(response.aswer), 2000):
                    response_copy = response.model_copy()
                    response_copy.answer = response.answer[i : i + 2000]
                    sent_message = await thread.send(
                        f"🤖 {format_response(response_copy, outro_message, lang_code)}",
                    )
            else:
                sent_message = await thread.send(
                    f"🤖 {format_response(response, outro_message, lang_code)}",
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
                return user == message.author and str(reaction.emoji) in ["👍", "👎"]

            try:
                reaction, user = await bot.wait_for("reaction_add", timeout=config.WAIT_TIME, check=check)

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
