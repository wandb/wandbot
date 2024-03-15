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
        self.bold_pattern = re.compile(r"\*\*([^*]+)\*\*", re.MULTILINE)
        self.strike_pattern = re.compile(r"~~([^~]+)~~", re.MULTILINE)
        self.header_pattern = re.compile(r"^#+\s*(.*?)\n", re.MULTILINE)

    @staticmethod
    def replace_markdown_link(match):
        text = match.group(1)
        url = match.group(2)
        return f"<{url}|{text}>"

    @staticmethod
    def replace_bold(match):
        return f"*{match.group(1)}*"

    @staticmethod
    def replace_strike(match):
        return f"~{match.group(1)}~"

    @staticmethod
    def replace_headers(match):
        header_text = match.group(1)
        return f"\n*{header_text}*\n"

    def __call__(self, text):
        try:
            segments = self.code_block_pattern.split(text)

            for i, segment in enumerate(segments):
                if segment.startswith("```") and segment.endswith("```"):
                    segment = self.language_spec_pattern.sub("```\n", segment)
                    segments[i] = segment
                else:
                    segment = self.markdown_link_pattern.sub(
                        self.replace_markdown_link, segment
                    )
                    segment = self.bold_pattern.sub(self.replace_bold, segment)
                    segment = self.strike_pattern.sub(
                        self.replace_strike, segment
                    )
                    segment = self.header_pattern.sub(
                        self.replace_headers, segment
                    )
                    segments[i] = segment

            return "".join(segments)
        except Exception:
            return text
