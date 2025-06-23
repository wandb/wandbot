"""Modal deployment for Wandbot bots only"""
import sys
from pathlib import Path

import modal
from modal import App, Image, Secret

# Add src to path for imports (go up one level from modal/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wandbot.configs.modal_config import ModalConfig
from wandbot.utils import get_logger

logger = get_logger(__name__)

# Load Modal configuration
modal_config = ModalConfig()

# Create a separate app for bots
bot_app = App("wandbot-bots")

# Build lightweight image for bots only
bot_image = (
    getattr(Image, modal_config.base_image)(python_version=modal_config.python_version)
    .apt_install("git", "curl", "ca-certificates")
    .pip_install("uv")
    .add_local_file("./pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_dir("./src", "/app/src", copy=True)
    .workdir("/app")
    # Install only bot dependencies using the optional dependency group
    .run_commands("uv pip install --system --compile-bytecode -e '.[bots]'")
    .env({"PYTHONPATH": "/app/src"})
)

# Reference Modal secrets
wandbot_secrets = Secret.from_name(modal_config.secrets_name)


@bot_app.function(
    image=bot_image,
    secrets=[wandbot_secrets],
    cpu=modal_config.bot_cpu,
    memory=modal_config.bot_memory,
    max_containers=1,  # Only one container for bots
    timeout=modal_config.bot_timeout,  # 24 hours
)
async def run_all_bots():
    """Run all bots in a single container"""
    import asyncio
    import os
    import sys
    import signal
    
    # Set the Modal API URL
    os.environ["WANDBOT_API_URL"] = "https://morg--wandbot-api-wandbotapi-serve.modal.run"
    sys.path.insert(0, "/app/src")
    
    # Track running tasks for graceful shutdown
    bot_tasks = []
    
    async def run_slack_bot_en():
        """Run English Slack bot"""
        while True:
            try:
                sys.argv = ['slack_bot', '-l', 'en']
                from wandbot.apps.slack.__main__ import main
                logger.info("🤖 Starting Slack bot (English)...")
                await main()
            except Exception as e:
                logger.error(f"Slack bot (English) crashed: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def run_slack_bot_ja():
        """Run Japanese Slack bot"""
        while True:
            try:
                sys.argv = ['slack_bot', '-l', 'ja']
                from wandbot.apps.slack.__main__ import main
                logger.info("🤖 Starting Slack bot (Japanese)...")
                await main()
            except Exception as e:
                logger.error(f"Slack bot (Japanese) crashed: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def run_discord_bot():
        """Run Discord bot"""
        while True:
            try:
                from wandbot.apps.discord.__main__ import main
                logger.info("🤖 Starting Discord bot...")
                await main()
            except Exception as e:
                logger.error(f"Discord bot crashed: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    # Handle graceful shutdown
    def signal_handler(_signum, _frame):
        logger.info("Received shutdown signal, cancelling bot tasks...")
        for task in bot_tasks:
            task.cancel()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run all bots concurrently
        logger.info("🚀 Starting all bots...")
        logger.info(f"📡 API URL set to: {os.environ.get('WANDBOT_API_URL')}")
        
        bot_tasks = [
            asyncio.create_task(run_slack_bot_en()),
            asyncio.create_task(run_slack_bot_ja()),
            asyncio.create_task(run_discord_bot())
        ]
        
        # Wait for all tasks (they should run forever unless cancelled)
        await asyncio.gather(*bot_tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Bot tasks cancelled, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error in run_all_bots: {e}")
        raise


# Schedule to keep bots running - runs immediately on deploy and then hourly
@bot_app.function(
    # schedule=modal.Cron("0 * * * *"),  # Every hour
    schedule=modal.Period(minutes=5),
    image=bot_image,
    secrets=[wandbot_secrets],
    cpu=0.25,  # Minimal resources for health check
    memory=512,
    max_containers=1,  # Only one container for bots
)
async def ensure_bots_running():
    """Check and restart bots if needed - also runs on initial deploy"""
    try:
        logger.info("🔄 Ensuring bots are running...")
        run_all_bots.spawn()
        logger.info("✅ Bot check completed")
    except Exception as e:
        logger.error(f"Error in ensure_bots_running: {e}")

# Ensure Modal finds the right app
app = bot_app