from typing import List

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from wandbot.database.client import DatabaseClient
from wandbot.utils import get_logger
from wandbot.youtube_chat.prompts import (
    keypoints_prompt_template,
    summaries_prompt_template,
)
from wandbot.youtube_chat.video_utils import YoutubeVideoInfoWithChapters

logger = get_logger(__name__)


class Summary(BaseModel):
    Missing_Entities: str = Field(
        ...,
        description="This is a list of 1-3 informative entities from the article which are missing from the previous summary",
    )
    Denser_Summary: str = Field(
        ...,
        description="This is a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities",
    )


class Summaries(BaseModel):
    summaries: List[Summary] = Field(
        ...,
        description="This is a list of 5 new, denser summaries of identical length which covers every entity and detail from the previous summary plus the missing entities",
        min_items=2,
        max_items=5,
    )


class KeyPoints(BaseModel):
    """
    This is a list of the most important points that were discussed or brought up in the video transcript.
    These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion.
    Your goal is to provide a list that someone could read to quickly understand what was talked about.
    """

    key_points: List[str] = Field(
        ...,
        description="This is a list of the most important points that were discussed or brought up in the video transcript. "
        "These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. "
        "Your goal is to provide a list that someone could read to quickly understand what was talked about.",
        min_items=3,
        max_items=10,
    )


def exception_to_messages(inputs: dict) -> dict:
    exception = inputs.pop("exception")
    messages = [
        BaseMessage(content=str(exception)),
        BaseMessage(
            content="The previous output raised an error. Please try again with corrected output."
        ),
    ]
    inputs["messages"] = messages
    return inputs


def get_summary_extraction_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summaries_prompt_template),
            ("human", "Video Transcript: {transcript}"),
            ("human", "!!! Tip: Make sure to answer in the correct format"),
        ]
    )
    model = ChatOpenAI(
        model_name="gpt-4-0125-preview",
        temperature=0.1,
        max_tokens=1000,
        max_retries=5,
    )
    model_with_structure = create_structured_output_runnable(
        Summaries, model, prompt
    )

    fallback_runnable = create_structured_output_runnable(
        output_schema=Summaries,
        llm=model,
        input_schema={"messages": [BaseMessage]},
    )

    self_correcting_chain = model_with_structure.with_fallbacks(
        [exception_to_messages | fallback_runnable], exception_key="exception"
    )

    self_correcting__retry_chain = self_correcting_chain.with_retry(
        stop_after_attempt=3
    )

    return self_correcting__retry_chain | RunnableLambda(
        lambda x: x.summaries[-1].Denser_Summary
    )


def get_key_points_extraction_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", keypoints_prompt_template),
            ("human", "Video Transcript: {transcript}"),
            ("human", "!!! Tip: Make sure to answer in the correct format"),
        ]
    )
    model = ChatOpenAI(
        model_name="gpt-4-0125-preview",
        temperature=0.1,
        max_tokens=1000,
        max_retries=5,
    )
    model_with_structure = create_structured_output_runnable(
        KeyPoints, model, prompt, mode="openai-tools"
    )

    fallback_runnable = create_structured_output_runnable(
        output_schema=KeyPoints,
        llm=model,
        input_schema={"messages": [BaseMessage]},
    )

    self_correcting_chain = model_with_structure.with_fallbacks(
        [exception_to_messages | fallback_runnable], exception_key="exception"
    )

    self_correcting__retry_chain = self_correcting_chain.with_retry(
        stop_after_attempt=3
    )

    return self_correcting__retry_chain | RunnableLambda(lambda x: x.key_points)


def get_summaries_and_keypoints_chain():
    summary_chain = get_summary_extraction_chain()
    key_points_chain = get_key_points_extraction_chain()
    full_summary_chain = RunnableParallel(
        summary=summary_chain, key_points=key_points_chain
    )
    return full_summary_chain


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
        self.video_info = YoutubeVideoInfoWithChapters(url=video_url)

    def transcribe(self):
        transcript = self.video_info.chapters_with_transcript
        # TODO: Add the transcript to the database
        # self.db_client.update_youtube_chat_thread(
        #     thread_id=self.thread

    def summarize(self):
        summary_chain = get_summary_extraction_chain()
        transcript_text = f"Title: {self.video_info.title}\n\nDescription: {self.video_info.description}\n\nTranscript: {self.video_info.transcript_text}"
        return summary_chain.invoke({"transcript": transcript_text})
