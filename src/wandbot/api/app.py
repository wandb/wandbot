import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from wandbot.api.routers import chat as chat_router
from wandbot.configs.app_config import AppConfig
from wandbot.database.database import engine
from wandbot.database.models import Base
from wandbot.utils import get_git_info, get_logger, log_disk_usage, log_top_disk_usage

app_config = AppConfig()

# Load environment variables from .env in project root
ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH, override=True)

logger = get_logger(__name__)

is_initialized = False
is_initializing = False


def _is_truthy(value: str | None, default: bool = False) -> bool:
    """Interpret common string values as booleans."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


async def initialize():
    global is_initialized, is_initializing
    print(f"STARTUP: initialize() function called - is_initialized: {is_initialized}, is_initializing: {is_initializing}")
    logger.info(
        f"STARTUP: initialize() function called - is_initialized: {is_initialized}, is_initializing: {is_initializing}"
    )

    if not is_initialized and not is_initializing:
        try:
            is_initializing = True
            logger.info("STARTUP: ⏳ Beginning initialization")

            # Check disk usage
            if os.getenv("LOG_LEVEL") == "DEBUG":
                disk_info = log_disk_usage()
                log_top_disk_usage()
                initial_disk_used = disk_info["used_mb"]
                logger.info(f"STARTUP: 💾 Initial disk usage: {initial_disk_used} MB")

            # 0/3: Initialize Weave
            weave_enabled = _is_truthy(os.getenv("WANDB_TRACING_ENABLED"), default=True)
            weave_required = _is_truthy(os.getenv("WANDB_TRACING_REQUIRED"))
            if weave_enabled:
                try:
                    logger.info("STARTUP: 0/3, Starting Weave...")
                    import weave

                    wandb_api_key = os.getenv("WANDB_API_KEY")
                    if not wandb_api_key:
                        logger.warning(
                            "STARTUP: 0/3, ⚠️ WANDB_API_KEY not set; skipping Weave initialization"
                        )
                    else:
                        import wandb

                        # Ensure wandb client is authenticated before initializing Weave
                        wandb.login(key=wandb_api_key, relogin=True)
                        weave.init(f"{app_config.wandb_entity}/{app_config.wandb_project}")
                        logger.info("STARTUP: 0/3, ✅ Weave initialized")
                except Exception as e:
                    logger.error(f"STARTUP: 0/3, ❌ Weave failed to initialize:{e}")
                    if weave_required:
                        raise
                    logger.warning("STARTUP: 0/3, ⚠️ Continuing without Weave tracing")
            else:
                logger.info("STARTUP: 0/3, ⏭️ Weave tracing disabled; skipping initialization")

            # Check disk usage
            if os.getenv("LOG_LEVEL") == "DEBUG":
                disk_info = log_disk_usage()
                log_top_disk_usage()
                disk_used_0 = disk_info["used_mb"]
                logger.info(
                    f"STARTUP: 0/3, 💾 Disk usage increment after 0: {round(disk_used_0 - initial_disk_used, 1)} MB"
                )

            # 1/3: Init Configs
            try:
                logger.info("STARTUP: 1/3, 📋 Initializing configs")
                from wandbot.chat.chat import ChatConfig
                from wandbot.configs.vector_store_config import VectorStoreConfig

                chat_config = ChatConfig()
                vector_store_config = VectorStoreConfig()

                chat_router.chat_components["chat_config"] = chat_config
                chat_router.chat_components["vector_store_config"] = vector_store_config

                # Log safe versions of configs
                safe_chat_config = {
                    k: v
                    for k, v in vars(chat_config).items()
                    if not any(sensitive in k.lower() for sensitive in ["key", "token"])
                }
                safe_vs_config = {
                    k: v
                    for k, v in vars(vector_store_config).items()
                    if not any(sensitive in k.lower() for sensitive in ["key", "token"])
                }

                logger.info(f"STARTUP: 1/3, 📋 Chat config: {str(safe_chat_config)}")
                logger.info(f"STARTUP: 1/3, 💿 Vector store config: {safe_vs_config}")
                logger.info("STARTUP: 1/3 ✅, 📋 Configs initialized")
            except Exception as e:
                logger.error("STARTUP: 1/3 ❌, 📋 Config initialization failed.")
                logger.error(f"STARTUP: 1/3 ❌, Error: {e}")
                raise

            # Check disk usage
            if os.getenv("LOG_LEVEL") == "DEBUG":
                disk_info = log_disk_usage()
                log_top_disk_usage()
                disk_used_1 = disk_info["used_mb"]
                logger.info(f"STARTUP: 1/3, 💾 Disk usage increment after 1: {round(disk_used_1 - disk_used_0, 1)} MB")

            # 2/3: Init Chat
            try:
                logger.info("STARTUP: 2/3, 💬 Starting Chat initialization")
                from wandbot.chat.chat import Chat

                chat_router.chat_components["chat"] = Chat(
                    vector_store_config=vector_store_config,
                    chat_config=chat_config,
                )
                logger.info("STARTUP: 2/3 ✅, 💬 Chat instance initialized.")
            except Exception as e:
                logger.error("STARTUP: 2/3 ❌, 💬 Chat instance initialization failed")
                logger.error(f"STARTUP: 2/3 ❌, Error: {e}")
                raise

            # 4/5: Init Retriever
            # try:
            #     logger.info(
            #         "STARTUP 4/5: ⚙️ Starting Retriever engine initialization"
            #     )
            #     from wandbot.api.routers import retrieve as retrieve_router

            #     retrieve_router.retriever = retrieve_router.SimpleRetrievalEngine(
            #         vector_store=vector_store,
            #         rerank_models={
            #             "english_reranker_model": chat_config.english_reranker_model,
            #             "multilingual_reranker_model": chat_config.multilingual_reranker_model,
            #         },
            #         chat_config=chat_router.chat_components["chat_config"],
            #     )
            #     logger.info("STARTUP 4/5: ✅ ⚙️ Retriever engine initialized")
            #     app.include_router(retrieve_router.router)
            #     logger.info("STARTUP 4/5: ✅ ⚙️ Added retrieve router to app.")
            # except Exception as e:
            #     logger.error(
            #         "STARTUP: 4/5 ❌, ⚙️ Retriever instance initializaion failed."
            #     )
            #     logger.error(f"STARTUP: 4/5 ❌, Error: {e}")
            #     raise

            # Check disk usage
            if os.getenv("LOG_LEVEL") == "DEBUG":
                disk_info = log_disk_usage()
                log_top_disk_usage()
                disk_used_2 = disk_info["used_mb"]
                logger.info(f"STARTUP: 2/3, 💾 Disk usage increment after 2: {round(disk_used_2 - disk_used_1, 1)} MB")

            # 3/3: Init Database
            try:
                Base.metadata.create_all(bind=engine, checkfirst=True)
                from wandbot.api.routers import database as database_router

                logger.info("STARTUP: 3/3, 🦉 Starting Database initialization")
                database_router.db_client = database_router.DatabaseClient()
                app.include_router(database_router.router)
                logger.info("STARTUP: 3/3, ✅ 🦉 Initialized database client")
            except Exception as e:
                logger.error("STARTUP: 3/3 ❌, 🦉 Database initialization failed.")
                logger.error(f"STARTUP: 3/3 ❌, Error: {e}")
                raise

            # Cleanup wandb artifacts cache
            logger.info("STARTUP: 3/3, 🧹 Cleaning up wandb artifacts cache")
            os.system("wandb artifact cache cleanup 0.01GB --remove-temp")

            # Check disk usage
            if os.getenv("LOG_LEVEL") == "DEBUG":
                disk_info = log_disk_usage()
                log_top_disk_usage()
                disk_used_3 = disk_info["used_mb"]
                logger.info(f"STARTUP: 3/3, 💾 Disk usage increment after 3: {round(disk_used_3 - disk_used_2, 1)} MB")

            is_initialized = True
            is_initializing = False
            logger.info("STARTUP: ✅ Initialization complete 🎉")
            if os.getenv("LOG_LEVEL") == "DEBUG":
                logger.info(
                    f"STARTUP: 💾 Total disk usage increment during initialization: {round(disk_used_3 - initial_disk_used, 1)} MB"
                )
            return {"startup_status": f"is_initialized: {is_initialized}"}

        except Exception as e:
            logger.error(f"STARTUP: 💀 Initialization failed: {e}")
            logger.error(f"STARTUP: 💀 Full error: {repr(e)}")
            raise

    else:
        logger.info(
            f"STARTUP: initialize() not started, is_initialized: {is_initialized}, is_initializing: {is_initializing}"
        )
        return {"startup_status": f"is_initialized: {is_initialized}, is_initializing: {is_initializing}"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for application startup and shutdown"""
    print("🔄 Starting initialization...")
    logger.info("🔄 Starting initialization...")
    await initialize()
    print("✅ Initialization complete!")
    logger.info("✅ Initialization complete!")
    yield
    print("🔄 Shutting down...")
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(title="Wandbot", version=app_config.wandbot_version or "1.3.3", lifespan=lifespan)
    
    # Configure all routes
    configure_routes(app)
    
    # Include routers
    app.include_router(chat_router.router)
    
    return app


def configure_routes(app: FastAPI) -> None:
    """Configure all application routes"""
    
    @app.get("/disk-usage")
    async def disk_usage_route():
        """
        Route to get disk usage information
        """
        return log_disk_usage()

    @app.get("/configs")
    async def configs():
        try:
            safe_chat_config = {
                k: v
                for k, v in vars(chat_router.chat_components["chat_config"]).items()
                if not any(sensitive in k.lower() for sensitive in ["key", "token"])
            }

            safe_vs_config = {
                k: v
                for k, v in vars(chat_router.chat_components["vector_store_config"]).items()
                if not any(sensitive in k.lower() for sensitive in ["key", "token"])
            }

            git_info = get_git_info()
            if all(v is None for v in git_info.values()):
                raise HTTPException(
                    status_code=500, detail="Unable to retrieve git information. Ensure this is a git repository."
                )
            git_info["timestamp"] = datetime.now().isoformat()

            return {
                "chat_config": safe_chat_config,
                "vector_store_config": safe_vs_config,
                "git_info": git_info,
                "app_config": app_config,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving configs: {str(e)}") from e

    @app.post("/initialize")
    async def initialize_endpoint():
        """Manual initialization endpoint for development with multiple workers"""
        result = await initialize()
        return result

    @app.get("/")
    @app.get("/status")
    async def status():
        global is_initialized, is_initializing
        try:
            from wandbot.api.routers import retrieve as retrieve_router
        except ImportError:
            retrieve_router = None

        chat_components = chat_router.chat_components

        c_ls = {}
        for key in chat_components.keys():
            c_ls[key] = str(type(chat_components[key]))

        return {
            "initialized": is_initialized,
            "initializing": is_initializing,
            "chat_ready": bool(chat_components.get("chat")),
            "retriever_ready": hasattr(retrieve_router, "retriever") if retrieve_router else False,
            "components": c_ls,
            "chat_type": str(type(chat_components.get("chat"))) if chat_components.get("chat") else None,
            "worker_info": {
                "process_id": os.getpid(),
                "note": "Each worker process has its own state when using --workers > 1"
            }
        }


# Create app instance
app = create_app()

def get_app() -> FastAPI:
    """Get the FastAPI application instance"""
    return app


# For running locally without Modal
if __name__ == "__main__":
    import uvicorn

    # Start the FastAPI server only (no bots)
    print("🚀 Starting Wandbot API on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("ℹ️  Note: This only starts the API server, not the Slack/Discord bots")
    print("   To run bots locally, see README.md for instructions")
    
    # Use the string import path for better reload support
    uvicorn.run(
        "wandbot.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info",
    )
