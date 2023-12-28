import regex as re


class MrkdwnFormatter:
    def __init__(self):
        self.code_block_pattern = re.compile(r"(```.*?```)", re.DOTALL)
        self.language_spec_pattern = re.compile(
            r"^```[a-zA-Z]+\n", re.MULTILINE
        )
        self.markdown_link_pattern = re.compile(
            r"\[([^\[]+)\]\((.*?)\)", re.MULTILINE
        )
        self.italic_pattern = re.compile(r"_([^_]+)_", re.MULTILINE)
        self.bold_pattern = re.compile(r"\*\*([^*]+)\*\*", re.MULTILINE)
        self.strike_pattern = re.compile(r"~~([^~]+)~~", re.MULTILINE)

    @staticmethod
    def replace_markdown_link(match):
        text = match.group(1)
        url = match.group(2)
        return f"<{url}|{text}>"

    @staticmethod
    def replace_italic(match):
        return f"*{match.group(1)}*"

    @staticmethod
    def replace_bold(match):
        return f"*{match.group(1)}*"

    @staticmethod
    def replace_strike(match):
        return f"~{match.group(1)}~"

    def __call__(self, text):
        try:
            # Split the text into segments based on code blocks
            segments = self.code_block_pattern.split(text)

            for i, segment in enumerate(segments):
                # If the segment is a code block, check for a language specification
                if segment.startswith("```") and segment.endswith("```"):
                    # Remove the language specification from the code block
                    segment = self.language_spec_pattern.sub("```\n", segment)
                    segments[i] = segment
                else:
                    # If the segment is not a code block, apply the conversion functions
                    segment = self.markdown_link_pattern.sub(
                        self.replace_markdown_link, segment
                    )
                    segment = self.italic_pattern.sub(
                        self.replace_italic, segment
                    )
                    segment = self.bold_pattern.sub(self.replace_bold, segment)
                    segment = self.strike_pattern.sub(
                        self.replace_strike, segment
                    )
                    segments[i] = segment

            # Join the segments back together
            return "".join(segments)
        except Exception:
            return text
