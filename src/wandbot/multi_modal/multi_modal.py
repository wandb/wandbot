import base64

from langchain.schema.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


class MultiModalQueryEngine:

    def __init__(self, model: str = "gpt-4-vision-preview"):
        self.model = ChatOpenAI(model_name=model)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
