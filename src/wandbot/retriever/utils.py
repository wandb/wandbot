from langchain_openai import OpenAIEmbeddings


class OpenAIEmbeddingsModel:
    def __init__(self):
        pass

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        model = OpenAIEmbeddings(
            model=value["embedding_model_name"],
            tiktoken_model_name=value["tokenizer_model_name"],
            dimensions=value["embedding_dimensions"],
        )
        setattr(obj, self.private_name, model)
