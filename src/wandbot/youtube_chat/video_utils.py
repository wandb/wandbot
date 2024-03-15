import os
import tempfile

from pytube import YouTube
from pytube.innertube import InnerTube
from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


def convert_duration_to_end(transcript):
    """
    Converts the 'duration' field in the transcript to an 'end' field.
    """
    for entry in transcript:
        entry["end"] = entry["start"] + entry["duration"]
    return transcript


def add_transcript_to_chapters(chapters, transcript):
    """
    Adds the relevant transcript entries to each chapter based on start and end times.
    """
    # First, convert the transcript durations to end times
    transcript = convert_duration_to_end(transcript)

    # Iterate through each chapter
    for chapter in chapters:
        # Filter transcript entries that fall within the chapter's start and end times
        chapter_transcript = [
            entry
            for entry in transcript
            if chapter["start"] <= entry["start"] <= chapter["end"]
        ]
        # Add these entries to the chapter under a new 'transcript' key
        chapter["transcript"] = chapter_transcript

    return chapters


class CustomInnerTube(InnerTube):
    def next(self, video_id):
        endpoint = f"{self.base_url}/next"
        query = {
            "videoId": video_id,
        }
        query.update(self.base_params)
        return self._call_api(endpoint, query, self.base_data)

    def get_transcript(self, params):
        endpoint = f"{self.base_url}/get_transcript"
        query = {
            "params": params,  # <-- !!!!!
        }
        query.update(self.base_params)
        result = self._call_api(endpoint, query, self.base_data)
        return result


def extract_transcript_params(next_data):
    engagement_panels = next_data["engagementPanels"]

    for engagement_panel in engagement_panels:
        engagement_panel_section = engagement_panel[
            "engagementPanelSectionListRenderer"
        ]

        if (
            engagement_panel_section.get("panelIdentifier")
            != "engagement-panel-searchable-transcript"
        ):
            continue

        return engagement_panel_section["content"]["continuationItemRenderer"][
            "continuationEndpoint"
        ]["getTranscriptEndpoint"]["params"]


class YoutubeVideoInfoWithChapters:
    def __init__(self, url: str):
        self.url = url
        self.youtube = YouTube(url)

        self.inner_tube = CustomInnerTube(client="WEB")

        self.video_id = self.youtube.video_id
        self.title = self.youtube.title
        self.embed_url = self.youtube.embed_url
        self.thumbnail_url = self.youtube.thumbnail_url
        self.author = self.youtube.author
        self.keywords = self.youtube.keywords
        self._chapters = None
        self._description = None
        self._transcript = None

    def get_chapters(self):
        data = self.inner_tube.next(self.video_id)
        params = extract_transcript_params(data)
        transcript = self.inner_tube.get_transcript(params)

        segments = transcript["actions"][0]["updateEngagementPanelAction"][
            "content"
        ]["transcriptRenderer"]["content"]["transcriptSearchPanelRenderer"][
            "body"
        ][
            "transcriptSegmentListRenderer"
        ][
            "initialSegments"
        ]

        chapters = []
        for segment in segments:
            if segment.get("transcriptSectionHeaderRenderer"):
                section = segment["transcriptSectionHeaderRenderer"]
                chapters.append(
                    dict(
                        start=float(section["startMs"]) / 1000.0,
                        end=float(section["endMs"]) / 1000.0,
                        label=section["snippet"]["simpleText"],
                    )
                )
        return chapters

    @property
    def chapters(self):
        if self._chapters is None:
            self._chapters = self.get_chapters()
        return self._chapters

    @property
    def description(self):
        if self._description is None:
            _ = self.youtube.streams.first()
            self._description = self.youtube.description
        return self._description

    @property
    def transcript(self):
        if self._transcript is None:
            transcript_list = YouTubeTranscriptApi.list_transcripts(
                self.video_id
            )
            try:
                transcript = transcript_list.find_manually_created_transcript(
                    ["en"]
                )
            except NoTranscriptFound:
                transcript = transcript_list.find_generated_transcript(["en"])
            self._transcript = transcript.fetch()

        return self._transcript

    @property
    def chapters_with_transcript(self):
        if self.chapters:
            chapters_with_transcript = add_transcript_to_chapters(
                self.chapters, self.transcript
            )
            return chapters_with_transcript
        else:
            return None

    @property
    def transcript_text(self):
        formatted_text = ""
        if self.chapters_with_transcript:
            for chapter in self.chapters_with_transcript:
                formatted_text += f"{chapter['label']}\n\n"
                for entry in chapter["transcript"]:
                    formatted_text += f"{entry['text']} "
                formatted_text += "\n\n"
            return formatted_text
        else:
            return (
                TextFormatter()
                .format_transcript(self.transcript)
                .replace("\n", " ")
            )

    @property
    def formatted_transcript(self):
        formatted_text = ""
        if self.chapters_with_transcript:
            for chapter in self.chapters_with_transcript:
                formatted_text += (
                    f"Chapter: {int(chapter['start'])}s {chapter['label']}\n\n"
                )
                for entry in chapter["transcript"]:
                    formatted_text += (
                        f"{int(entry['start'])}s: {entry['text']}\n"
                    )
                formatted_text += "\n\n"
            return formatted_text
        else:
            for entry in self.transcript:
                formatted_text += f"{int(entry['start'])}s: {entry['text']}\n"
            return formatted_text

    @property
    def transcript_citations(self):
        citations = {}
        for entry in self.transcript:
            citations[int(entry["start"])] = entry["text"]
        return citations

    @property
    def audio(self):
        # Use tempfile to create a temp directory, ensuring our temp file has a .mp3 extension
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "audio.mp3")
            youtube = YouTube(self.url)

            # Download and convert the audio stream to MP3 in the temp file
            stream = youtube.streams.filter(only_audio=True).first()
            stream.download(filename=temp_path)

            # Open the temporary MP3 file in binary mode to read its bytes
            with open(temp_path, "rb") as temp_file:
                audio_bytes = temp_file.read()

        # Return the bytes of the MP3 audio file
        return audio_bytes
