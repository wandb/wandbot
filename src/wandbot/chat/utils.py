import re
from typing import Any, Dict, List

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain.schema import Document

from wandbot.database.schemas import QuestionAnswer


def get_chat_history(
    chat_history: List[QuestionAnswer] | None,
) -> List[tuple[str, str]]:
    if not chat_history:
        return []
    else:
        return [
            (question_answer.question, question_answer.answer)
            for question_answer in chat_history
        ]


class ConversationalRetrievalQASourcesChain(ConversationalRetrievalChain):
    reduce_k_below_max_tokens: bool = True
    max_tokens_limit: int = 2816

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
            self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        return self._reduce_tokens_below_limit(docs)

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"] + ["sources"]
        return _output_keys

    def _call(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        results = super()._call(inputs, **kwargs)
        answer = results["answer"]
        if re.search(r"Sources:\s", answer, flags=re.IGNORECASE):
            answers_and_sources = re.split(r"Sources:\s", answer, flags=re.IGNORECASE)
            if len(answers_and_sources) > 1:
                answer = answers_and_sources[0]
                sources = answers_and_sources[1]
            elif len(answers_and_sources) == 1:
                answer = answers_and_sources[0]
                sources = ""
            else:
                sources = ""
        else:
            sources = ""
        results["answer"] = answer
        results["sources"] = sources
        return results
