from io import BytesIO
from typing import Any, Dict, List

from pydantic import AfterValidator, BaseModel, Field, HttpUrl, computed_field
from pytube import YouTube
from typing_extensions import Annotated
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter

HttpUrlString = Annotated[HttpUrl, AfterValidator(str)]


class YoutubeVideoInfo(BaseModel):
    url: HttpUrlString = Field(..., title="The URL of the YouTube video")

    @computed_field
    @property
    def embed_url(self) -> str:
        return YouTube(self.url).embed_url

    @computed_field
    @property
    def author(self) -> str | None:
        return YouTube(self.url).author

    @computed_field
    @property
    def thumbnail_url(self) -> str:
        return YouTube(self.url).thumbnail_url

    @computed_field
    @property
    def video_id(self) -> str:
        return YouTube(self.url).video_id

    @computed_field
    @property
    def title(self) -> str | None:
        return YouTube(self.url).title

    @computed_field
    @property
    def description(self) -> str | None:
        return YouTube(self.url).description

    @computed_field
    @property
    def keywords(self) -> List[str] | None:
        return YouTube(self.url).keywords


class YoutubeVideoInfoWithTranscript(YoutubeVideoInfo):
    @computed_field
    @property
    def transcript(self) -> List[Dict[str, Any]]:
        return YouTubeTranscriptApi().get_transcript(
            self.video_id, preserve_formatting=True
        )

    @computed_field
    @property
    def transcript_bytes(self) -> bytes:
        return SRTFormatter().format_transcript(self.transcript).encode("utf-8")


class YoutubeVideoInforWithAudio(YoutubeVideoInfo):
    @computed_field
    @property
    def audio(self) -> bytes:
        buffer = BytesIO()
        YouTube(self.url).streams.filter(
            only_audio=True
        ).first().stream_to_buffer(buffer)
        return buffer.getvalue()
