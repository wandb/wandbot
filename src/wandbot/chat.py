import wandb
from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import FAISS

PROJECT = "wandb_docs_bot"

run = wandb.init(project=PROJECT)


def load_artifacts():
    faiss_artifact = run.use_artifact(
        "parambharat/wandb_docs_bot/faiss_store:latest", type="search_index"
    )
    faiss_artifact_dir = faiss_artifact.download()

    hyde_prompt_artifact = run.use_artifact(
        "parambharat/wandb_docs_bot/hyde_prompt:latest", type="prompt"
    )
    hyde_artifact_dir = hyde_prompt_artifact.download()
    hyde_prompt_file = f"{hyde_artifact_dir}/hyde_prompt.txt"

    chat_prompt_artifact = run.use_artifact(
        "parambharat/wandb_docs_bot/system_prompt:latest", type="prompt"
    )
    chat_artifact_dir = chat_prompt_artifact.download()
    chat_prompt_file = f"{chat_artifact_dir}/chat_prompt.txt"

    return {
        "faiss": faiss_artifact_dir,
        "hyde_prompt": hyde_prompt_file,
        "chat_prompt": chat_prompt_file,
    }


def load_hyde_prompt(fname):
    prompt_template = open(fname).read()
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
    faiss_store = FAISS.load_local(store_dir, embeddings)
    return faiss_store


def load_chat_prompt(fname):
    prompt_template = open(fname).read()

    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def load_qa_chain():
    artifacts = load_artifacts()
    vectorstore = load_faiss_store(
        artifacts["faiss"],
        artifacts["hyde_prompt"],
    )
    chat_prompt = load_chat_prompt(artifacts["chat_prompt"])
    chain = VectorDBQAWithSourcesChain.from_chain_type(
        ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
        ),
        chain_type="stuff",
        vectorstore=vectorstore,
        chain_type_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
    )
    return chain


def get_answer(chain, question):
    query = " ".join(question.strip().split())
    result = chain(
        {
            "question": query,
        },
        return_only_outputs=True,
    )
    sources = ["- " + doc.metadata["source"] for doc in result["source_documents"]]
    response = result["answer"] + "\n\n*References*:\n\n>" + "\n>".join(sources)
    return response


class Chat:
    def __init__(
        self,
    ):
        self.qa_chain = load_qa_chain()

    def __call__(self, query):
        response = get_answer(self.qa_chain, query)
        return response
