from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from wandbot.utils import get_logger
from wandbot.api.routers import chat as chat_router
from wandbot.database.database import engine
from wandbot.database.models import Base
import os
import shutil
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path)

logger = get_logger(__name__)

is_initialized = False
is_initializing = False


def get_disk_usage() -> Dict:
    """
    Get disk usage information and return it as a dictionary.
    Can be used both as a helper function and route handler.
    """
    try:
        total, used, free = shutil.disk_usage("/")
        current_dir = Path(".")
        current_dir_size = sum(
            f.stat().st_size for f in current_dir.glob("**/*") if f.is_file()
        )
        
        # Calculate values in GB
        total_gb = round(total / (2**30), 2)
        used_gb = round(used / (2**30), 2)
        used_mb = round(used / (2**20), 2)
        free_gb = round(free / (2**30), 2)
        current_dir_gb = round(current_dir_size / (2**30), 2)
        usage_percentage = round((used * 100 / total), 2)
        
        # Create response dictionary
        disk_info = {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "used_mb": used_mb,
            "free_gb": free_gb,
            "current_dir_gb": current_dir_gb,
            "usage_percentage": usage_percentage
        }
        
        # Log the information
        logger.info(f"DISK USAGE: 💾 Total Disk Size: {total_gb} GB")
        logger.info(f"DISK USAGE: 📊 Used Space: {used_gb} GB")
        logger.info(f"DISK USAGE: ✨ Free Space: {free_gb} GB")
        logger.info(f"DISK USAGE: 📂 Current Directory Size: {current_dir_gb} GB")
        logger.info(f"DISK USAGE: 💯 Disk Usage Percentage: {usage_percentage}%")
        
        return disk_info
        
    except Exception as e:
        error_msg = f"❌ Error getting disk usage: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


async def initialize():
    global is_initialized, is_initializing
    logger.info(
        f"STARTUP: initialize() function called, \nis_initialized: {is_initialized} is_initializing: {is_initializing}"
    )

    if not is_initialized and not is_initializing:
        try:
            is_initializing = True
            logger.info("STARTUP: ⏳ Beginning initialization")

            # Check disk usage
            try:
                disk_info = get_disk_usage()
                initial_disk_used = disk_info['used_mb']
                logger.info(f"STARTUP: 💾 Initial disk usage: {initial_disk_used} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise

            # 0/5: Initalise Weave
            try:
                logger.info("STARTUP: 0/5, Starting Weave...")
                import weave

                weave.init(
                    f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
                )
                logger.info("STARTUP: 0/5, ✅ Weave initialized")
            except Exception as e:
                logger.error(
                    f"STARTUP: 0/5, ❌ Weave failed to initialize:\n{e}"
                )
                raise

            # Check disk usage
            try:
                disk_info = get_disk_usage()
                disk_used_0 = disk_info['used_mb']
                logger.info(f"STARTUP: 0/5, 💾 Disk usage increment after 0: {round(disk_used_0 - initial_disk_used, 1)} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise
            
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
            try:
                disk_info = get_disk_usage()
                disk_used_1 = disk_info['used_mb']
                logger.info(f"STARTUP: 1/5, 💾 Disk usage increment after 1: {round(disk_used_1 - disk_used_0, 1)} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise

            # 2/5: Init Vector store config
            try:
                logger.info("STARTUP: 2/5, 💿 Initializing vector store")
                from wandbot.retriever import VectorStore
                from wandbot.ingestion.config import VectorStoreConfig

                vector_store_config = VectorStoreConfig()
                chat_router.chat_components[
                    "vector_store_config"
                ] = vector_store_config
                safe_vs_config = {k: v for k, v in vars(vector_store_config).items() 
                             if not any(sensitive in k.lower() for sensitive in ['key', 'token'])}
                logger.info(
                    f"STARTUP: 2/5, Vector store config: {safe_vs_config}"
                )

                vector_store = VectorStore.from_config(vector_store_config)
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
            try:
                disk_info = get_disk_usage()
                disk_used_2 = disk_info['used_mb']
                logger.info(f"STARTUP: 2/5, 💾 Disk usage increment after 2: {round(disk_used_2 - disk_used_1, 1)} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise
            
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
            try:
                disk_info = get_disk_usage()
                disk_used_3 = disk_info['used_mb']
                logger.info(f"STARTUP: 3/5, 💾 Disk usage increment after 3: {round(disk_used_3 - disk_used_2, 1)} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise
            
            # 4/5: Init Retriever
            try:
                logger.info(
                    "STARTUP 4/5: ⚙️ Starting Retriever engine initialization"
                )
                from wandbot.api.routers import retrieve as retrieve_router

                retrieve_router.retriever = retrieve_router.SimpleRetrievalEngine(
                    vector_store=vector_store,
                    rerank_models={
                        "english_reranker_model": chat_config.english_reranker_model,
                        "multilingual_reranker_model": chat_config.multilingual_reranker_model,
                    },
                )
                logger.info("STARTUP 4/5: ✅ ⚙️ Retriever engine initialized")
                app.include_router(retrieve_router.router)
                logger.info("STARTUP 4/5: ✅ ⚙️ Added retrieve router to app.")
            except Exception as e:
                logger.error(
                    "STARTUP: 4/5 ❌, ⚙️ Retriever instance initializaion failed."
                )
                logger.error(f"STARTUP: 4/5 ❌, Error: {e}")
                raise

            # Check disk usage
            try:
                disk_info = get_disk_usage()
                disk_used_4 = disk_info['used_mb']
                logger.info(f"STARTUP: 4/5, 💾 Disk usage increment after 4: {round(disk_used_4 - disk_used_3, 1)} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise

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
            
            # Check disk usage
            try:
                disk_info = get_disk_usage()
                disk_used_5 = disk_info['used_mb']
                logger.info(f"STARTUP: 5/5, 💾 Disk usage increment after 5: {round(disk_used_5 - disk_used_4, 1)} MB")
                if "error" in disk_info:
                    logger.error(f"STARTUP: -- ❌, {disk_info['error']}")
            except Exception as e:
                logger.error(
                    f"STARTUP: -- ❌, Get disk usage failed, error: {e}"
                )
                raise

            is_initialized = True
            is_initializing = False
            logger.info("STARTUP: ✅ Initialization complete 🎉")
            logger.info(f"STARTUP: 💾 Total disk usage increment during intialization: {round(disk_used_5 - initial_disk_used, 1)} MB")
            return {"startup_status": f"is_initialized: {is_initialized}"}

        except Exception as e:
            logger.error(f"STARTUP: 💀 Initialization failed:\n{e}")
            logger.error(f"STARTUP: 💀 Full error:\n{repr(e)}")
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
    return get_disk_usage()


@app.get("/status")
async def status():
    global is_initialized, is_initializing
    try:
        from wandbot.api.routers import retrieve as retrieve_router
    except ImportError:
        retrieve_router = None

    components = chat_router.chat_components
    config = components.get("vector_store_config")

    c_ls = {}
    for key in components.keys():
        c_ls[key] = str(type(components[key]))

    return {
        "initialized": is_initialized,
        "initializing": is_initializing,
        "chat_ready": bool(components.get("chat")),
        "retriever_ready": hasattr(retrieve_router, "retriever")
        if retrieve_router
        else False,
        "vector_store_ready": bool(components.get("vector_store")),
        "components": c_ls,
        "chat_type": str(type(components.get("chat")))
        if components.get("chat")
        else None,
        "vector_store_config": {
            "persist_dir": str(config.persist_dir) if config else None,
            "collection_name": config.collection_name if config else None,
        }
        if config
        else None,
    }


@app.get("/")
async def root(background_tasks: BackgroundTasks):
    global is_initializing, is_initialized
    return {
        "is_initializing": is_initializing,
        "is_initialized": is_initialized,
        "message": "Hello.",
    }


app.include_router(chat_router.router)
