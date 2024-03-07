from datetime import timedelta
from typing import List, Literal

from openai import OpenAI
from openai.types import FileObject
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run, ThreadMessage
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from wandbot.database.client import DatabaseClient
from wandbot.database.schemas import YoutubeAssistantThreadCreate
from wandbot.utils import get_logger

logger = get_logger(__name__)


from wandbot.youtube_chat.video_utils import YoutubeVideoInfoWithTranscript

ASSISTANT_INSTRUCTIONS = "You are a AI designed to specifically chat with youtube videos. Use the transcript in your knowledge base to best respond to user queries."


class Message(BaseModel):
    role: Literal["assistant", "user"]
    content: str


def is_run_in_progress(run: Run):
    """Return True if value is None"""
    logger.info(f"Run status: {run.status}")
    return run.status in ("queued", "in_progress")


class YoutubeAssistant:
    def __init__(
        self,
        thread_id: str,
        video_url: str,
        db_client: DatabaseClient,
        model="gpt-4-turbo-preview",
    ):
        self.thread_id = thread_id
        self.video_url = video_url
        self.db_client = db_client
        self.model = model
        self.video_info = YoutubeVideoInfoWithTranscript(url=video_url)
        self.client = OpenAI()
        self.files: List[FileObject] = []
        self.assistant: Assistant | None = None
        self.thread: Thread | None = None
        self.run: Run | None = None
        self.instructions: str = ASSISTANT_INSTRUCTIONS

    def __enter__(self):
        thread_details = self.db_client.get_youtube_assistant_thread(
            self.thread_id
        )
        logger.info(f"Thread details: {thread_details}")
        if thread_details is None:
            file = self.client.files.create(
                file=self.video_info.transcript_bytes, purpose="assistants"
            )
            self.files.append(file)
            self.assistant = self.client.beta.assistants.create(
                instructions=self.instructions,
                model=self.model,
                tools=[{"type": "retrieval"}],
                file_ids=[file.id for file in self.files],
            )
            self.thread = self.client.beta.threads.create()

            thread_details = self.db_client.create_youtube_assistant_thread(
                YoutubeAssistantThreadCreate(
                    thread_id=self.thread_id,
                    video_id=self.video_info.video_id,
                    assistant_file_id=file.id,
                    assistant_id=self.assistant.id,
                    assistant_thread_id=self.thread.id,
                )
            )
            logger.info(f"Created youtube assistant thread: {thread_details}")
            return self
        file = self.client.files.retrieve(thread_details.assistant_file_id)
        self.files.append(file)
        self.assistant = self.client.beta.assistants.retrieve(
            thread_details.assistant_id
        )
        self.thread = self.client.beta.threads.retrieve(
            thread_details.assistant_thread_id
        )

        return self

    def cleanup(self, time_delta: timedelta = timedelta(days=1)):
        deleted_threads = self.db_client.delete_old_youtube_assistant_threads(
            time_delta=time_delta
        )
        logger.info(f"Deleted threads: {deleted_threads}")
        if deleted_threads:
            for deleted_thread in deleted_threads:
                self.client.beta.assistants.files.delete(
                    assistant_id=deleted_thread.assistant_id,
                    file_id=deleted_thread.assistant_file_id,
                )
                self.client.files.delete(
                    file_id=deleted_thread.assistant_file_id
                )
                self.client.beta.assistants.delete(deleted_thread.assistant_id)
                self.client.beta.threads.delete(
                    deleted_thread.assistant_thread_id
                )

    def __exit__(self, exc_type, exc_value, traceback):
        update_youtube_assistant_thread = (
            self.db_client.update_youtube_assistant_thread_time(self.thread_id)
        )
        logger.info(
            f"Updated youtube assistant thread: {update_youtube_assistant_thread}"
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_result(is_run_in_progress),
        stop=stop_after_attempt(10),
    )
    def retrieve_run(self, run_id: str) -> Run:
        return self.client.beta.threads.runs.retrieve(
            thread_id=self.thread.id, run_id=run_id
        )

    def format_message(self, message: ThreadMessage) -> str:
        if getattr(message.content[0], "text", None) is not None:
            message_content = message.content[0].text
        else:
            message_content = message.content[0]
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f" [{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = self.client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] {file_citation.quote} from {cited_file.filename}"
                )
            elif file_path := getattr(annotation, "file_path", None):
                cited_file = self.client.files.retrieve(file_path.file_id)
                citations.append(
                    f"[{index}] file: {cited_file.filename} is downloaded"
                )

        # message_content.value += "\n" + "\n".join(citations)
        return message_content.value

    def extract_run_message(self) -> str:
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
        ).data
        for message in messages:
            if message.run_id == self.run.id:
                return f"{message.role}: " + self.format_message(
                    message=message
                )
        return "Assistant: No message found"

    def chat(self, messages: List[Message]) -> str:
        for message in messages:
            _ = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=message.role,  # type: ignore
                content=message.content,
            )

        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id, assistant_id=self.assistant.id
        )

        self.run = self.retrieve_run(self.run.id)

        if self.run.status == "failed":
            raise Exception(
                f"Run failed with the following error {self.run.last_error}"
            )
        if self.run.status == "expired":
            raise Exception(f"Run expired when calling {self.run}")
        return self.extract_run_message()
