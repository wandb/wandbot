import time
from typing import List, Any, Dict

import wandb
from langchain import LLMChain
from langchain.chains import (
    HypotheticalDocumentEmbedder,
    ConversationalRetrievalChain,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever


class VectorStoreRetrieverWithScore(VectorStoreRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
            docs = []
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
                docs.append(doc)
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


class FAISSWithScore(FAISS):
    def as_retriever(self) -> VectorStoreRetrieverWithScore:
        return VectorStoreRetrieverWithScore(
            vectorstore=self,
            search_type="similarity",
            search_kwargs={"k": 10},
        )


class ConversationalRetrievalQAWithSourcesChainWithScore(ConversationalRetrievalChain):
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

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        # question = inputs[self.question_key]
        docs = self.retriever.get_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)


def load_artifacts(config):
    faiss_artifact = wandb.use_artifact(config.faiss_artifact, type="search_index")
    faiss_artifact_dir = faiss_artifact.download()

    hyde_prompt_artifact = wandb.use_artifact(
        config.hyde_prompt_artifact, type="prompt"
    )
    hyde_artifact_dir = hyde_prompt_artifact.download()
    hyde_prompt_file = f"{hyde_artifact_dir}/hyde_prompt.txt"

    # chat_prompt_artifact = wandb.use_artifact(
    #     config.chat_prompt_artifact, type="prompt"
    # )
    # chat_artifact_dir = chat_prompt_artifact.download()
    # chat_prompt_file = f"{chat_artifact_dir}/chat_prompt.txt"

    return {
        "faiss": faiss_artifact_dir,
        "hyde_prompt": hyde_prompt_file,
        "chat_prompt": None,  # chat_prompt_file,
    }


def load_hyde_prompt(f_name):
    prompt_template = open(f_name).read()
    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def load_hyde_embeddings(prompt_file, temperature=0.3):
    prompt = load_hyde_prompt(prompt_file)
    base_embeddings = OpenAIEmbeddings()
    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=LLMChain(llm=ChatOpenAI(temperature=temperature), prompt=prompt),
        base_embeddings=base_embeddings,
    )
    return embeddings


def load_faiss_store(store_dir, prompt_file, temperature=0.3):
    embeddings = load_hyde_embeddings(prompt_file, temperature)
    faiss_store = FAISSWithScore.load_local(store_dir, embeddings)
    return faiss_store


def load_chat_prompt(f_name):
    prompt_template = open(f_name).read()

    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def load_vector_store_and_prompt(config):
    artifacts = load_artifacts(config)
    vector_store = load_faiss_store(
        artifacts["faiss"],
        artifacts["hyde_prompt"],
    )
    chat_prompt = load_chat_prompt(
        "/media/mugan/data/wandb/projects/wandbot/src/artifacts/system_prompt:v0/chat_prompt.txt"
    )

    return vector_store, chat_prompt


def load_qa_chain(model_name="gpt-4", vector_store=None, chat_prompt=None):
    chain = ConversationalRetrievalQAWithSourcesChainWithScore.from_llm(
        ChatOpenAI(
            model_name=model_name,
            temperature=0,
        ),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        qa_prompt=chat_prompt,
        return_source_documents=True,
    )
    return chain


def get_answer(chain, question, chat_history=[]):
    query = " ".join(question.strip().split())
    result = chain(
        {
            "question": query,
            "chat_history": chat_history,
        },
        return_only_outputs=True,
    )
    sources = list(
        {
            "- " + doc.metadata["source"]
            for doc in result["source_documents"]
            if doc.metadata["score"] <= 0.4
        }
    )

    if len(sources):
        response = result["answer"]  # + "\n\n*References*:\n\n" + "\n".join(sources)
    else:
        response = result["answer"]
    return response


class Chat:
    def __init__(
        self,
        model_name="gpt-4",
        wandb_run=None,
    ):
        self.model_name = model_name
        self.wandb_run = wandb_run

        self.settings = ""
        for k, v in wandb_run.config.as_dict().items():
            self.settings + f"{k}:{v}\n"

        self.vector_store, self.chat_prompt = load_vector_store_and_prompt(
            config=wandb_run.config
        )
        self.qa_chain = load_qa_chain(
            model_name="gpt-4",
            vector_store=self.vector_store,
            chat_prompt=self.chat_prompt,
        )

    def __call__(self, query, chat_history=[]):
        start_time = time.time()
        # Try call GPT-4, if not fall back to 3.5 turbo
        try:
            response = get_answer(self.qa_chain, query, chat_history)
        except:
            print("Falling back to gpt-3.5-turbo")
            self.qa_chain = load_qa_chain(
                model_name="gpt-3.5-turbo",
                vector_store=self.vector_store,
                chat_prompt=self.chat_prompt,
            )
            response = get_answer(self.qa_chain, query)
            fallback_warning = "**Warning: Falling back to gpt-3.5.** These results are sometimes not as good as gpt-4"
            response = fallback_warning + "\n\n" + response

        end_time = time.time()
        elapsed_time = end_time - start_time
        timings = (start_time, end_time, elapsed_time)
        # To consider, add verbose mode
        # if "--v" or "--verbose" in query:
        #     return query, response + "\n\n" + self.settings, timings
        return query, response + "\n\n", timings


def main():
    from wandbot.config import default_config

    run = wandb.init(project="wandb_docs_bot_dev", config=default_config.__dict__)
    run.config.update(default_config.__dict__)
    chat = Chat(model_name=run.config.model_name, wandb_run=run)
    chat_history = []
    while True:
        user_query = input("Enter your question:")
        query, response, time = chat(user_query)
        chat_history.append((user_query, response))
        print(response)


if __name__ == "__main__":
    main()
