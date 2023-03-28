import wandb
from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import FAISS

def load_artifacts():
    faiss_artifact = run.use_artifact(
      config.faiss_artifact, type="search_index"
    )
    faiss_artifact_dir = faiss_artifact.download()

    hyde_prompt_artifact = run.use_artifact(
      config.hyde_prompt_artifact, type="prompt"
    )
    hyde_artifact_dir = hyde_prompt_artifact.download()
    hyde_prompt_file = f"{hyde_artifact_dir}/hyde_prompt.txt"

    chat_prompt_artifact = run.use_artifact(
      config.chat_prompt_artifact, type="prompt"
    )
    chat_artifact_dir = chat_prompt_artifact.download()
    chat_prompt_file = f"{chat_artifact_dir}/chat_prompt.txt"

    return {
        "faiss": faiss_artifact_dir,
        "hyde_prompt": hyde_prompt_file,
        "chat_prompt": chat_prompt_file,
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
    faiss_store = FAISS.load_local(store_dir, embeddings)
    return faiss_store


def load_chat_prompt(f_name):
    prompt_template = open(f_name).read()

    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def load_qa_chain(model_name="gpt-4"):
    artifacts = load_artifacts()
    vector_store = load_faiss_store(
        artifacts["faiss"],
        artifacts["hyde_prompt"],
    )
    chat_prompt = load_chat_prompt(artifacts["chat_prompt"])
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(
            model_name=model_name,
            temperature=0,
        ),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
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
    response = result["answer"] # + "\n\n*References*:\n\n>" + "\n>".join(sources)
    return response


class Chat:
    def __init__(
        self,
        model_name="gpt-4",
        wandb_run=None,
    ):
        self.model_name = model_name
        self.qa_chain = load_qa_chain(self.model_name)

        self.settings = ""
        if wandb_run is not None:
            self.wandb_run_id = wandb_run.id
            config = wandb_run.config.to_dict()
            for k,v in config: 
                self.settings + f"{k}:{v}\n"
        


    def __call__(self, query):
        # Try call GPT-4, if not fall back to 3.5
        response = get_answer(self.qa_chain, query)
        if "--v" or "--verbose" in query:
          return response + "\n\n" + self.settings
        else:
          return response + "\n\n"
        # if "xxx" not in response:
        #   return response + "\n" + "gpt-3.5-turbo\n\n"
        # else:
        #   self.chain = load_qa_chain("gpt-3.5-turbo")
        #   response = get_answer(self.qa_chain, query)
        #   return response + "\n\n" + f"powered-by: {self.model_name}\n\n"
          
def main():
    chat = Chat(
      model_name=run.config.model_name, 
      config=run.config.to_dict()
      wandb_run=run,
    )
    user_query = input("Enter your question:")
    response = chat(user_query)
    print(response)

if __name__ == "__main__":
    main()