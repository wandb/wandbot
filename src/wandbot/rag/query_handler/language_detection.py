from operator import itemgetter

from langchain_core.runnables import Runnable, RunnablePassthrough

from wandbot.utils import FastTextLangDetect, FasttextModelConfig


class LanguageDetector:
    model_config = FasttextModelConfig()

    def __init__(self):
        self.model = FastTextLangDetect(self.model_config)
        self._chain = None

    @property
    def chain(self) -> Runnable:
        lang_detect_chain = RunnablePassthrough().assign(
            language=lambda x: self.model.detect_language(x["question"])
        ) | itemgetter("language")
        return lang_detect_chain
