from operator import itemgetter

from langchain_core.runnables import Runnable, RunnablePassthrough

from wandbot.utils import FastTextLangDetect, FasttextModelConfig


class LangDetect:
    def __init__(self, model):
        self.model = FastTextLangDetect(
            FasttextModelConfig(
                fasttext_file_path=model,
            )
        )

    def __call__(self, question: str) -> str:
        return self.model.detect_language(question)


def load_language_detection_chain(model: str) -> Runnable:
    lang_detect = LangDetect(model)
    lang_detect_chain = RunnablePassthrough().assign(
        language=lambda x: lang_detect(x["question"])
    ) | itemgetter("language")
    return lang_detect_chain
