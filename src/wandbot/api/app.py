from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from wandbot.utils import get_logger, log_disk_usage, log_top_disk_usage, get_git_info
from wandbot.api.routers import chat as chat_router
from wandbot.database.database import engine
from wandbot.database.models import Base
import os
from datetime import datetime
from dotenv import load_dotenv

from wandbot.configs.app_config import AppConfig

app_config = AppConfig()

dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path, override=True)

logger = get_logger(__name__)

is_initialized = False
is_initializing = False


async def initialize():
    global is_initialized, is_initializing
    logger.info(
        f"STARTUP: initialize() function called, is_initialized: {is_initialized} is_initializing: {is_initializing}"
    )

    if not is_initialized and not is_initializing:
        try:
            is_initializing = True
            logger.info("STARTUP: ⏳ Beginning initialization")

            # Check disk usage
            disk_info = log_disk_usage()
            log_top_disk_usage()
            initial_disk_used = disk_info['used_mb']
            logger.info(f"STARTUP: 💾 Initial disk usage: {initial_disk_used} MB")

            # 0/5: Initalise Weave
            try:
                logger.info("STARTUP: 0/5, Starting Weave...")
                import weave

                weave.init(
                    f"{app_config.wandb_entity}/{app_config.wandb_project}"
                )
                # Undo autopatch for langchain for now as its very verbose
                from weave.integrations.langchain.langchain import langchain_patcher
                langchain_patcher.undo_patch()
                logger.info("STARTUP: 0/5, ✅ Weave initialized")
            except Exception as e:
                logger.error(
                    f"STARTUP: 0/5, ❌ Weave failed to initialize:{e}"
                )
                raise

            # Check disk usage
            disk_info = log_disk_usage()
            log_top_disk_usage()
            disk_used_0 = disk_info['used_mb']
            logger.info(f"STARTUP: 0/5, 💾 Disk usage increment after 0: {round(disk_used_0 - initial_disk_used, 1)} MB")
            
            # 1/5: Init Chat config
            try:
                logger.info("STARTUP: 1/5, 📋 Init Chat config")
                from wandbot.chat.chat import ChatConfig

                chat_config = ChatConfig()
                chat_router.chat_components["chat_config"] = chat_config
                logger.info("STARTUP: 1/5 ✅, 📋 Chat config initialized")
            except Exception as e:
                logger.error(
                    "STARTUP: 1/5 ❌, 📋 Vector store initialization failed."
                )
                logger.error(f"STARTUP: 1/5 ❌, Error: {e}")
                raise

            # Check disk usage
            disk_info = log_disk_usage()
            log_top_disk_usage()
            disk_used_1 = disk_info['used_mb']
            logger.info(f"STARTUP: 1/5, 💾 Disk usage increment after 1: {round(disk_used_1 - disk_used_0, 1)} MB")

            # 2/5: Init Vector store config
            try:
                logger.info("STARTUP: 2/5, 💿 Initializing vector store")
                
                from wandbot.configs.vector_store_config import VectorStoreConfig
                vector_store_config = VectorStoreConfig()
                chat_router.chat_components[
                    "vector_store_config"
                ] = vector_store_config
                safe_vs_config = {k: v for k, v in vars(vector_store_config).items() 
                             if not any(sensitive in k.lower() for sensitive in ['key', 'token'])}
                logger.info(
                    f"STARTUP: 2/5, Vector store config: {safe_vs_config}"
                )

                from wandbot.retriever import VectorStore
                vector_store = VectorStore.from_config(
                    vector_store_config=chat_router.chat_components["vector_store_config"],
                    chat_config=chat_router.chat_components["chat_config"]
                )
                chat_router.chat_components["vector_store"] = vector_store
                logger.info(
                    "STARTUP: 2/5 ✅, 💿 Vector store created successfully."
                )
            except Exception as e:
                logger.error(
                    "STARTUP: 2/5 ❌, 💿 Vector store initialization failed."
                )
                logger.error(f"STARTUP: 2/5 ❌, Error: {e}")
                logger.error(
                    f"STARTUP: 2/5 ❌, 💿 Vector store config details: {vars(vector_store_config)}"
                )
                raise
            
            # Check disk usage
            disk_info = log_disk_usage()
            log_top_disk_usage()
            disk_used_2 = disk_info['used_mb']
            logger.info(f"STARTUP: 2/5, 💾 Disk usage increment after 2: {round(disk_used_2 - disk_used_1, 1)} MB")
            
            # 3/5: Init Chat
            try:
                logger.info("STARTUP: 3/5, 💬 Starting Chat initialization")
                safe_chat_config = {k: v for k, v in vars(chat_router.chat_components['chat_config']).items() 
                             if not any(sensitive in k.lower() for sensitive in ['key', 'token'])}
                logger.info(
                    f"STARTUP: 3/5, 💬 Chat config to be used: {str(safe_chat_config)}"
                )
                from wandbot.chat.chat import Chat

                chat_router.chat_components["chat"] = Chat(
                    vector_store=chat_router.chat_components["vector_store"],
                    config=chat_router.chat_components["chat_config"],
                )
                logger.info("STARTUP: 3/5 ✅, 💬 Chat instance initialized.")
            except Exception as e:
                logger.error(
                    "STARTUP: 3/5 ❌, 💬 Chat instance initializaion failed"
                )
                logger.error(f"STARTUP: 3/5 ❌, Error: {e}")
                raise
            
            # Check disk usage
            disk_info = log_disk_usage()
            log_top_disk_usage()
            disk_used_3 = disk_info['used_mb']
            logger.info(f"STARTUP: 3/5, 💾 Disk usage increment after 3: {round(disk_used_3 - disk_used_2, 1)} MB")

            
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
            disk_info = log_disk_usage()
            log_top_disk_usage()
            disk_used_4 = disk_info['used_mb']
            logger.info(f"STARTUP: 4/5, 💾 Disk usage increment after 4: {round(disk_used_4 - disk_used_3, 1)} MB")

            # 5/5: Init Database
            try:
                Base.metadata.create_all(bind=engine)
                from wandbot.api.routers import database as database_router

                logger.info("STARTUP: 5/5, 🦉 Starting Database initialization")
                database_router.db_client = database_router.DatabaseClient()
                app.include_router(database_router.router)
                logger.info("STARTUP: 5/5, ✅ 🦉 Initialized database client")
            except Exception as e:
                logger.error("STARTUP: 5/5 ❌, 🦉 Databse initializaion failed.")
                logger.error(f"STARTUP: 5/5 ❌, Error: {e}")
                raise

            # Cleanup wandb artifacts cache
            logger.info("STARTUP: 5/5, 🧹 Cleaning up wandb artifacts cache")
            os.system("wandb artifact cache cleanup 0.01GB --remove-temp")
            
            # Check disk usage
            disk_info = log_disk_usage()
            log_top_disk_usage()
            disk_used_5 = disk_info['used_mb']
            logger.info(f"STARTUP: 5/5, 💾 Disk usage increment after 5: {round(disk_used_5 - disk_used_4, 1)} MB")

            is_initialized = True
            is_initializing = False
            logger.info("STARTUP: ✅ Initialization complete 🎉")
            logger.info(f"STARTUP: 💾 Total disk usage increment during intialization: {round(disk_used_5 - initial_disk_used, 1)} MB")
            return {"startup_status": f"is_initialized: {is_initialized}"}

        except Exception as e:
            logger.error(f"STARTUP: 💀 Initialization failed: {e}")
            logger.error(f"STARTUP: 💀 Full error: {repr(e)}")
            raise

    else:
        logger.info(
            f"STARTUP: initialize() not started, is_initialized: {is_initialized}, is_initializing: {is_initializing}"
        )
        return {
            "startup_status": f"is_initialized: {is_initialized}, is_initializing: {is_initializing}"
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running preliminary setup...")
    if os.getenv("WANDBOT_FULL_INIT"):
        logger.info("Running full wandbot initialization...")
        await initialize()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Wandbot", version="1.3.0", lifespan=lifespan)


@app.get("/startup")
async def startup():
    global is_initialized, is_initializing

    if is_initialized:
        logger.info("✅ Startup already complete.")
        return {"status": "already_initialized"}

    if is_initializing:
        logger.info("⏳ Startup initialization already in progress...")
        return {"status": "initializing"}

    try:
        logger.info("📦 Main startup initialization triggered.")
        _ = await initialize()
        return {"status": "initializing"}
    except Exception as e:
        logger.error(f"💀 Startup initialization failed: {str(e)}")
        is_initializing = False
        return {"status": "initialization_failed", "error": str(e)}


@app.get("/disk-usage")
async def disk_usage_route():
    """
    Route to get disk usage information
    """
    return log_disk_usage()


@app.get("/configs")
async def configs():
    try:
        safe_chat_config = {k: v for k, v in vars(chat_router.chat_components['chat_config']).items() 
                if not any(sensitive in k.lower() for sensitive in ['key', 'token'])}
    
        safe_vs_config = {k: v for k, v in vars(chat_router.chat_components['vector_store_config']).items() 
                if not any(sensitive in k.lower() for sensitive in ['key', 'token'])}
    
        git_info = get_git_info()
        if all(v is None for v in git_info.values()):
            raise HTTPException(
                status_code=500,
                detail="Unable to retrieve git information. Ensure this is a git repository."
            )
        git_info["timestamp"] = datetime.now().isoformat()

        return {
            "chat_config": safe_chat_config, 
            "vector_store_config": safe_vs_config,
            "git_info": git_info
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving configs: {str(e)}"
        )


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
        "retriever_ready": hasattr(retrieve_router, "retriever")
        if retrieve_router
        else False,
        "vector_store_ready": bool(chat_components.get("vector_store")),
        "components": c_ls,
        "chat_type": str(type(chat_components.get("chat")))
        if chat_components.get("chat")
        else None,
    }


app.include_router(chat_router.router)
