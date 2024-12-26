"""This module serves as the main server API for the wandbot application.

It imports and uses the FastAPI framework to define the API and initialize application event handlers like "startup".
Also, the module includes Python's built-in asyncio library for managing asynchronous tasks related to database backup.

The API includes:
- APICreateChatThreadRequest
- APIFeedbackRequest
- APIFeedbackResponse
- APIGetChatThreadResponse
- APIQueryRequest
- APIQueryResponse
- APIQuestionAnswerRequest
- APIQuestionAnswerResponse

Following classes and their functionalities:
- Chat: Main chat handling class, initialized during startup.
- ChatConfig: Configuration utility for chat.
- ChatRequest: Schema to handle requests made to the chat.

It also sets up and interacts with the database through:
- DatabaseClient: A utility to interact with the database.
- Base.metadata.create_all(bind=engine): Creates database tables based on the metadata.

The server runs periodic backup of the data to wandb using the backup_db method which runs as a coroutine.
The backup data is transformed into a Pandas DataFrame and saved as a wandb.Table.

It uses logger from the utils module for logging purposes.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from wandbot.utils import get_logger

from wandbot.api.routers import chat as chat_router

from wandbot.database.database import engine
from wandbot.database.models import Base

logger = get_logger(__name__)

is_initialized = False
is_initializing = False

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from wandbot.utils import get_logger

logger = get_logger(__name__)

is_initialized = False
is_initializing = False

async def initialize():
    """
    Startup initialization - Imports and intializations are done lazily so as to not block the startup of the application
    """
    global is_initialized, is_initializing
    logger.info(f"STARTUP: initialize() function called,\
\nis_initialized: {is_initialized}\
\is_initializing: {is_initializing}")
    
    if not is_initialized and not is_initializing:
        try:
            is_initializing = True
            logger.info("STARTUP: ⏳ Beginning initialization")

            try:
                logger.info("STARTUP: 0/5, Starting Weave...")
                import weave
                weave.init(f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}")
                logger.info("STARTUP: 0/5, ✅ Weave initialized")
            except Exception as e:
                logger.error(f"STARTUP: 0/5, ❌ Weave failed to initialize:\n{e}")
                

            # Initialize Chat config
            try:
                logger.info("STARTUP: 1/5, 📋 Init Chat config")
                from wandbot.chat.chat import ChatConfig
                chat_config = ChatConfig()
                chat_router.chat_components["chat_config"] = chat_config
                logger.info("STARTUP: 1/5 ✅, 📋 Chat config initialized")
            except Exception as e:
                logger.error(f"STARTUP: 1/5 ❌, 📋 Vector store initialization failed.")
                logger.error(f"STARTUP: 1/5 ❌, Error: {e}")
                raise
    
            # Initialize vector store
            try:
                logger.info("STARTUP: 2/5, 💿 Initializing vector store")
                from wandbot.retriever import VectorStore
                from wandbot.ingestion.config import VectorStoreConfig
                vector_store_config = VectorStoreConfig()
                chat_router.chat_components["vector_store_config"] = vector_store_config
                logger.info(f"STARTUP: 2/5, Vector store config: {vector_store_config}")
                
                vector_store = VectorStore.from_config(vector_store_config)
                chat_router.chat_components["vector_store"] = vector_store
                logger.info("STARTUP: 2/5 ✅, 💿 Vector store created successfully.")
            except Exception as e:
                logger.error(f"STARTUP: 2/5 ❌, 💿 Vector store initialization failed.")
                logger.error(f"STARTUP: 2/5 ❌, Error: {e}")
                logger.error(f"STARTUP: 2/5 ❌, 💿 Vector store config details: {vars(vector_store_config)}")
                raise
    
            # Initialize Chat
            try:
                logger.info("STARTUP: 3/5, 💬 Starting Chat initialization")
                logger.info(f"STARTUP: 3/5, 💬 Chat config to be used: {str(chat_router.chat_components['chat_config'])}")
                from wandbot.chat.chat import Chat
                chat_router.chat_components["chat"] = Chat(
                    vector_store=chat_router.chat_components["vector_store"],
                    config=chat_router.chat_components["chat_config"]
                )
                logger.info("STARTUP: 3/5 ✅, 💬 Chat instance initialized.")
                # print(f"✨ Chat instance created")
            except Exception as e:
                logger.error("STARTUP: 3/5 ❌, 💬 Chat instance initializaion failed")
                logger.error(f"STARTUP: 3/5 ❌, Error: {e}")
                raise
        
            # Initialize Retriever
            try:
                logger.info("STARTUP 4/5: ⚙️ Starting Retriever engine initialization")
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
                logger.error("STARTUP: 4/5 ❌, ⚙️ Retriever instance initializaion failed.")
                logger.error(f"STARTUP: 4/5 ❌, Error: {e}")
                raise

            # Initialize Database
            try:
                Base.metadata.create_all(bind=engine)
                from wandbot.api.routers import database as database_router
                logger.info("STARTUP: 5/5, 🦉 Starting Database initialization")
                database_router.db_client = database_router.DatabaseClient()
                app.include_router(database_router.router)
                logger.info(f"STARTUP: 5/5, ✅ 🦉 Initialized database client")
            except Exception as e:
                logger.error("STARTUP: 5/5 ❌, 🦉 Databse initializaion failed.")
                logger.error(f"STARTUP: 5/5 ❌, Error: {e}")

            is_initialized = True
            is_initializing = False
            logger.info("STARTUP: ✅ Initialization complete 🎉")
            return {"startup_status": f"is_initialized: {is_initialized}"}
    
        except Exception as e:
            logger.error(f"STARTUP: 💀 Initialization failed:\n{e}")
            logger.error(f"STARTUP: 💀 Full error:\n{repr(e)}")
            raise
         
    else:
        logger.info(f"STARTUP: initialize() not started, is_initialized: {is_initialized}, is_initializing: {is_initializing}")
        return {"startup_status": f"is_initialized: {is_initialized}, is_initializing: {is_initializing}"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The main initialization gets tiggered in the /startup route. For some reason
    running `await initialize()`, `asyncio.create_task(initialize())` or 
    `background_tasks.add_task(initialize)` does not work when called within the `lifespan` conext manager.
    """
    logger.info("Running preliminary setup...")

    # is_initializing = True
    # init_task = asyncio.create_task(initialize())
    # await initialize()
    # await initialize_heavy_components()
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="Wandbot",
    version="1.3.0",
    lifespan=lifespan
)

@app.get("/startup")
async def startup():
    """Trigger startup of remaining chat and retriever engines."""
    global is_initialized, is_initializing

    if is_initialized:
        logger.info("✅ Startup already complete.")
        return {"status": "already_initialized"}

    if is_initializing:
        logger.info("⏳ Startup initialization already in progress...")
        return {"status": "initializing"}

    # Tigger initialization
    try:
        logger.info("📦 Main startup initialization triggered.")
        _  = await initialize()
        return {"status": "initializing"}
    except Exception as e:
        logger.error(f"💀 Startup initialization failed: {str(e)}")
        is_initializing = False
        return {"status": "initialization_failed", "error": str(e)}

# @app.get("/startup")
# async def startup():
#     """Trigger startup of remaining chat and retriever engines"""
#     global is_initialized, is_initializing
#     if not is_initialized and not is_initializing:
#         try:
#             logger.info("📦 Startup initialization triggered")
#             is_initializing = True
#             await initialize()
#         except Exception as e:
#             logger.info(f"💀 Startup initialization failed: {str(e)}") 
                # raise
#     elif is_initializing and not is_initialized:
#         logger.info("⏳ Startup initialization already in progress")
#     elif is_initialized:
#         logger.info("✅ Startup already complete")         


@app.get("/status")
async def status():
    """Detailed status endpoint""" 
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
        "retriever_ready": hasattr(retrieve_router, "retriever") if retrieve_router else False,
        "vector_store_ready": bool(components.get("vector_store")),
        "components": c_ls,
        "chat_type": str(type(components.get("chat"))) if components.get("chat") else None,
        "vector_store_config": {
            "persist_dir": str(config.persist_dir) if config else None,
            "collection_name": config.collection_name if config else None
        } if config else None
    }

@app.get("/")
async def root(background_tasks: BackgroundTasks):
    """Used by Replit as part of a health check"""
    global is_initializing, is_initialized

    # if is_initialized:
    #     return {"status": "ready", "message": "Wandbot is initialized and running."}

    # if is_initializing:
    #     return {"status": "initializing", "message": "Wandbot is currently initializing."}

    # is_initializing = True
    # await initialize()
    # background_tasks.add_task(initialize)
    return {
        "is_initializing": is_initializing,
        "is_initialized" : is_initialized,
        "message": "Hello."
    }

app.include_router(chat_router.router)