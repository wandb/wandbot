"""Modal deployment configuration for Wandbot"""
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

# Create the Modal app
modal_app = App(modal_config.app_name)

# Build image with optimized settings
wandbot_image = (
    getattr(Image, modal_config.base_image)(python_version=modal_config.python_version)
    .apt_install("git", "curl", "ca-certificates")
    .pip_install("uv")
    .add_local_file("../pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_dir("../src", "/app/src", copy=True)
    .workdir("/app")
    .run_commands("uv pip install --system --compile-bytecode -e .")
    .env({"PYTHONPATH": "/app/src"})
)

# Reference Modal secrets
wandbot_secrets = Secret.from_name(modal_config.secrets_name)


@modal_app.cls(
    image=wandbot_image,
    secrets=[wandbot_secrets],
    cpu=modal_config.api_cpu,
    memory=modal_config.api_memory,
    min_containers=modal_config.api_min_containers,
    max_containers=modal_config.api_max_containers,
    timeout=modal_config.api_timeout,
)
@modal.concurrent(max_inputs=modal_config.api_max_inputs)
class WandbotAPI:
    @modal.enter()
    async def start_services(self):
        """Initialize services when container starts"""
        logger.info("🚀 Starting API service...")
        
        # Initialize the FastAPI app
        from wandbot.api.app import get_app
        logger.info("🔄 Container starting - getting Wandbot app...")
        self.app = get_app()
        logger.info("✅ Container ready to serve requests")
    
    @modal.asgi_app()
    def serve(self):
        """
        Wandbot API Modal deployment
        """
        return self.app




# For local testing with Modal
@modal_app.local_entrypoint()
def main():
    logger.info("🚀 Wandbot API - Modal Deployment")
    logger.info("")
    logger.info("This deployment includes:")
    logger.info("  • API Server: The FastAPI application")
    logger.info("")
    logger.info("Deploy with:")
    logger.info("  modal deploy modal/modal_app.py")
    logger.info("")
    logger.info("For bots, deploy separately:")
    logger.info("  modal deploy modal/modal_bots.py")




# Ensure Modal finds the right app
app = modal_app