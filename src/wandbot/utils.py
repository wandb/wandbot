"""This module contains utility functions and classes for the Wandbot system.

The module includes the following functions:
- `get_logger`: Creates and returns a logger with the specified name.
- `load_embeddings`: Loads embeddings from cache or creates new ones if not found.
- `load_llm`: Loads a language model with the specified parameters.
- `load_service_context`: Loads a service context with the specified parameters.
- `load_storage_context`: Loads a storage context with the specified parameters.
- `load_index`: Loads an index from storage or creates a new one if not found.

The module also includes the following classes:
- `Timer`: A simple timer class for measuring elapsed time.

Typical usage example:

    logger = get_logger("my_logger")
    embeddings = load_embeddings("/path/to/cache")
    llm = load_llm("gpt-3", 0.5, 3)
    service_context = load_service_context(llm, 0.5, "/path/to/cache", 3)
    storage_context = load_storage_context(768, "/path/to/persist")
    index = load_index(nodes, service_context, storage_context, "/path/to/persist")
"""

import asyncio
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import sqlite3
import string
import subprocess
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import fasttext
import nest_asyncio
import tiktoken
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

import wandb
from wandbot.schema.document import Document


def get_logger(name: str) -> logging.Logger:
    """Creates and returns a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance with the specified name.
    """

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Get log level from environment or default to INFO
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = level_map.get(log_level, logging.INFO)  # Default to INFO if invalid level

    # Get the logger instance
    logger = logging.getLogger(name)
    
    # Set the logger level explicitly
    logger.setLevel(level)
    
    # Only add handler if the logger doesn't already have handlers
    if not logger.handlers:
        # Configure rich console with custom theme
        theme = Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "critical": "red bold",
            "debug": "grey50"
        })
        console = Console(theme=theme, width=130, tab_size=4)
        
        # Create and configure the handler
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
            omit_repeated_times=True
        )
        handler.setFormatter(logging.Formatter("WANDBOT | %(message)s"))
        handler.setLevel(level)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


logger = get_logger(__name__)


def strip_punctuation(text):
    # Create a translation table mapping every punctuation character to None
    translator = str.maketrans("", "", string.punctuation)

    # Use the table to strip punctuation from the text
    no_punct = text.translate(translator)
    return no_punct


class Timer:
    """A simple timer class for measuring elapsed time."""

    def __init__(self) -> None:
        """Initializes the timer."""
        self.start = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.stop = self.start

    def __enter__(self) -> "Timer":
        """Starts the timer."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Stops the timer."""
        self.stop = datetime.datetime.now().astimezone(datetime.timezone.utc)

    @property
    def elapsed(self) -> float:
        """Calculates the elapsed time in seconds."""
        return (self.stop - self.start).total_seconds()


def cachew(cache_path: str = "./cache.db", logger=None):
    """
    Memoization decorator that caches the output of a method in a SQLite
    database.
    ref: https://www.kevinkatz.io/posts/memoize-to-sqlite
    """
    db_conn = sqlite3.connect(cache_path)
    db_conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, result TEXT)"
    )

    def memoize(func):
        def wrapped(*args, **kwargs):
            # Compute the hash of the <function name>:<argument>
            xs = f"{func.__name__}:{repr(tuple(args))}:{repr(kwargs)}".encode(
                "utf-8"
            )
            arg_hash = hashlib.sha256(xs).hexdigest()

            # Check if the result is already cached
            cursor = db_conn.cursor()
            cursor.execute(
                "SELECT result FROM cache WHERE hash = ?", (arg_hash,)
            )
            row = cursor.fetchone()
            if row is not None:
                if logger is not None:
                    logger.debug(
                        f"Cached result found for {arg_hash}. Returning it."
                    )
                return json.loads(row[0])

            # Compute the result and cache it
            result = func(*args, **kwargs)
            cursor.execute(
                "INSERT INTO cache (hash, result) VALUES (?, ?)",
                (arg_hash, json.dumps(result)),
            )
            db_conn.commit()

            return result

        return wrapped

    return memoize


class FasttextModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    fasttext_file_path: pathlib.Path = pathlib.Path(
        "data/cache/models/lid.176.bin"
    )
    fasttext_artifact_path: str = Field(
        "wandbot/wandbot_public/fasttext-lid.176.bin:v0",
        env="LANGDETECT_ARTIFACT_PATH",
        validation_alias="langdetect_artifact_path",
    )
    fasttext_artifact_type: str = "fasttext-model"
    wandb_project: str = Field(
        "wandbot-dev", env="WANDB_PROJECT", validation_alias="wandb_project"
    )
    wandb_entity: str = Field(
        "wandbot", env="WANDB_ENTITY", validation_alias="wandb_entity"
    )


class FastTextLangDetect:
    """Uses fasttext to detect the language of a text, from this file:
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    """

    def __init__(self, config: FasttextModelConfig = FasttextModelConfig()):
        self.config = config
        self._model = self._load_model()

    def detect_language(self, text: str):
        cleaned_text = strip_punctuation(text).replace("\n", " ")
        predictions = self.model.predict(cleaned_text)
        return predictions[0][0].replace("__label__", "")

    def detect_language_batch(self, texts: List[str]):
        predictions = self.model.predict(texts)
        return [p[0].replace("__label__", "") for p in predictions[0]]

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        if not os.path.isfile(self.config.fasttext_file_path):
            if wandb.run is None:
                api = wandb.Api()
                artifact = api.artifact(self.config.fasttext_artifact_path)
            else:
                artifact = wandb.run.use_artifact(
                    self.config.fasttext_artifact_path,
                    type=self.config.fasttext_artifact_type,
                )
            _ = artifact.download(
                root=str(self.config.fasttext_file_path.parent)
            )
        self._model = fasttext.load_model(str(self.config.fasttext_file_path))
        return self._model


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> Tuple[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: List[Any] = tasks

    nest_asyncio.apply()
    if show_progress:
        try:
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one

            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[Any]:
                return await tqdm.gather(
                    *tasks_to_execute, desc=progress_bar_desc
                )

            tqdm_outputs: Tuple[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> Tuple[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: Tuple[Any] = asyncio.run(_gather())
    return outputs


def clean_document_content(doc: Document) -> Document:
    cleaned_content = re.sub(r"\n{3,}", "\n\n", doc.page_content)
    cleaned_content = cleaned_content.strip()
    cleaned_document = Document(
        page_content=cleaned_content, metadata=doc.metadata
    )
    cleaned_document = make_document_tokenization_safe(cleaned_document)
    return cleaned_document


def make_document_tokenization_safe(document: Document) -> Document:
    """Removes special tokens from the given documents.

    Args:
        documents: A list of strings representing the documents.

    Returns:
        A list of cleaned documents with special tokens removed.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    special_tokens_set = encoding.special_tokens_set

    def remove_special_tokens(text: str) -> str:
        """Removes special tokens from the given text.

        Args:
            text: A string representing the text.

        Returns:
            The text with special tokens removed.
        """
        for token in special_tokens_set:
            text = text.replace(token, "")
        return text

    content = document.page_content
    cleaned_document = remove_special_tokens(content)
    return Document(page_content=cleaned_document, metadata=document.metadata)


def filter_smaller_documents(
    documents: List[Document], min_size: int = 3, min_line_size: int = 5
) -> List[Document]:
    def filter_small_document(document: Document) -> bool:
        return (
            len(
                [
                    doc
                    for doc in document.page_content.split("\n")
                    if len(doc.strip().split()) >= min_line_size
                ]
            )
            >= min_size
        )

    return [
        document for document in documents if filter_small_document(document)
    ]


def log_disk_usage(dir: str = ".") -> Dict:
    """
    Get disk usage information and return it as a dictionary.
    Can be used both as a helper function and route handler.
    """
    try:
        total, used, free = shutil.disk_usage(dir)
        current_dir = Path(dir)
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
        logger.info(f"DISK USAGE: Total Disk Size: {total_gb} GB")
        logger.info(f"DISK USAGE: Used Space: {used_gb} GB")
        logger.info(f"DISK USAGE: Free Space: {free_gb} GB")
        logger.info(f"DISK USAGE: Current Directory Size: {current_dir_gb} GB")
        logger.info(f"DISK USAGE: Disk Usage Percentage: {usage_percentage}%")
        
        return disk_info
        
    except Exception as e:
        error_msg = f"❌ Error getting disk usage: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def log_top_disk_usage(dir: str = ".", top_n: int = 20):
    try:
        logger.info("STARTUP: --, 📂 Getting top 20 files/directories by disk usage")
        import subprocess
        command = f"du -ah {dir} | sort -rh | head -n {top_n}"
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"STARTUP: --, 📂 Top {top_n} files/directories by disk usage:\n{result.stdout}\n")
        else:
            logger.error(f"STARTUP: -- ❌, Failed to get disk usage, error: {result.stderr}")
    except Exception as e:
        logger.error(f"STARTUP: -- ❌, Error getting top {top_n} files/directories by disk usage: {e}")


def run_git_command(command: list[str]) -> Optional[str]:
    """
    Runs a git command and returns the output.
    Returns None if the command fails.
    """
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return None

def get_git_info() -> Dict[str, Optional[str]]:
    """
    Retrieves comprehensive git information about the current repository.
    Returns a dictionary with the git information or None values if commands fail.
    """
    info = {}
    
    # Basic repository info
    info["commit_hash"] = run_git_command(['git', 'rev-parse', 'HEAD'])
    info["commit_hash_short"] = run_git_command(['git', 'rev-parse', '--short', 'HEAD'])
    info["branch"] = run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    
    # Get remote URL and repository name
    remote_url = run_git_command(['git', 'config', '--get', 'remote.origin.url'])
    if remote_url:
        info["remote_url"] = remote_url
        if remote_url.endswith('.git'):
            remote_url = remote_url[:-4]
        info["repository"] = remote_url.split('/')[-1]
    else:
        info["remote_url"] = None
        info["repository"] = None
    
    # Commit details
    info["last_commit_date"] = run_git_command(['git', 'log', '-1', '--format=%cd', '--date=iso'])
    info["last_commit_author"] = run_git_command(['git', 'log', '-1', '--format=%an'])
    info["last_commit_message"] = run_git_command(['git', 'log', '-1', '--format=%s'])
    
    # Repository state
    info["is_dirty"] = run_git_command(['git', 'status', '--porcelain']) != ""
    
    # Tags
    latest_tag = run_git_command(['git', 'describe', '--tags', '--abbrev=0'])
    info["latest_tag"] = latest_tag
    
    if latest_tag:
        # Commits since last tag
        commits_since_tag = run_git_command([
            'git', 'rev-list', f'{latest_tag}..HEAD', '--count'
        ])
        info["commits_since_tag"] = commits_since_tag
    
    # Get total number of commits
    info["total_commits"] = run_git_command(['git', 'rev-list', '--count', 'HEAD'])
    
    # Get configured user info
    info["config_user_name"] = run_git_command(['git', 'config', 'user.name'])
    
    # Remote tracking info
    tracking_branch = run_git_command(['git', 'rev-parse', '--abbrev-ref', '@{upstream}'])
    if tracking_branch:
        info["tracking_branch"] = tracking_branch
        
        # Get ahead/behind counts
        ahead_behind = run_git_command([
            'git', 'rev-list', '--left-right', '--count', f'HEAD...{tracking_branch}'
        ])
        if ahead_behind:
            ahead, behind = ahead_behind.split('\t')
            info["commits_ahead"] = ahead
            info["commits_behind"] = behind
    
    return info

def run_sync(coro: Coroutine) -> Any:
    """
    Safely run an async coroutine in a synchronous context.
    If no event loop is running, we create one. 
    Otherwise, we schedule the coroutine on the existing loop.

    Args:
        coro: The coroutine to run synchronously

    Returns:
        The result of the coroutine execution

    Example:
        In practice, from inside a FastAPI endpoint that is already async,
        you'd typically do:
           await chat_instance.__acall__(request)
        But if you have a pure sync path, you can do:
           result = run_sync(chat_instance.__acall__(request))
    """
    try:
        loop = asyncio.get_running_loop()
        # If we get here, it means there's already a running loop. 
        # We'll schedule the coroutine thread-safely:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    except RuntimeError:
        # No running loop, so create our own
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

def get_error_file_path(tb) -> str:
    """Extract the file path where an error occurred from a traceback.
    
    Args:
        tb: A traceback object from sys.exc_info()[2]
        
    Returns:
        The file path where the error occurred
    """
    try:
        while tb.tb_next is not None:
            tb = tb.tb_next
        return tb.tb_frame.f_code.co_filename
    except Exception as e:
        logger.error(f"Error getting error file path from traceback: {e}")
        return None

class ErrorInfo(BaseModel):
    """Base model for error information that can be included in any response"""
    has_error: bool = False
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    component: Optional[str] = None  # e.g. "reranker", "embedding", "llm"
    stacktrace: Optional[str] = None  # Full stacktrace when error occurs
    file_path: Optional[str] = None  # File where the error occurred
