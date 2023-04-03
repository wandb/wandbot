import argparse
import json
import logging
import os
import pathlib
from typing import Dict, List, Union, Optional

import langchain
import pandas as pd
import wandb
from langchain import LLMChain, FAISS
from langchain.cache import SQLiteCache
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import (
    UnstructuredMarkdownLoader,
    NotebookLoader,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter,
)
from tqdm import tqdm

from wandbot.prompts import load_hyde_prompt

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)


def create_qa_prompt(df):
    new_df = df.apply(
        lambda x: f"Question:\n{'-' * 10}\n{x['question']}\n\nAnswer:\n{'-' * 10}\n{x['answer']}",
        axis=1,
    )
    new_df = pd.DataFrame(new_df, columns=["reference"])
    new_df["source"] = df["source"]
    return new_df.to_dict(orient="records")


def load_csv_data(f_name):
    df = pd.read_csv(f_name)
    if "title" in df.columns:
        df["question"] = df["title"] + "\n\n" + df["question"]
    if "source" not in df.columns:
        df["source"] = f"{f_name}-" + df.index.map(str)
    return create_qa_prompt(df)


def map_git_to_local_paths(paths: List[str], examples=True) -> Dict[str, str]:
    local_paths = list(map(lambda x: str(x), paths))
    if examples:
        git_paths = map(lambda x: "/".join(x.split("/")[1:]), local_paths)
        git_paths = map(
            lambda x: f"https://github.com/wandb/examples/blob/master/{x}", git_paths
        )
    else:
        git_paths = map(lambda x: "/".join(x.split("/")[3:]), local_paths)
        git_paths = map(
            lambda x: f"https://github.com/wandb/wandb/blob/main/{x}", git_paths
        )
    return dict(zip(local_paths, git_paths))


def load_notebook_paths(notebook_dir: str = "examples/colabs/") -> Dict[str, str]:
    paths = pathlib.Path(notebook_dir).rglob("*.ipynb*")
    return map_git_to_local_paths(paths)


def load_code_paths(
    code_dir: str = "examples/examples/", examples=True
) -> Dict[str, str]:
    paths = pathlib.Path(code_dir).rglob("*.py*")
    return map_git_to_local_paths(paths, examples=examples)


def load_documentation_paths(docs_dir: str = "docodile") -> Dict[str, str]:
    paths = pathlib.Path(docs_dir).rglob("*.md*")
    paths = filter(lambda x: "readme" not in str(x).lower(), paths)
    path_parts = map(lambda x: x.parts, paths)
    path_parts = list(filter(lambda x: len(x) > 2, path_parts))
    git_paths = map(lambda x: str(pathlib.Path(*x)), path_parts)

    link_paths = map(lambda x: pathlib.Path(*x[2:]), path_parts)
    link_paths = map(
        lambda x: str(x.parent / "" if "intro" in x.stem else x.stem), link_paths
    )

    link_paths = map(lambda x: f"https://docs.wandb.ai/{x}", link_paths)
    return dict(zip(git_paths, link_paths))


def map_source(documents: List[Document], source_map: Dict[str, str]) -> List[Document]:
    for document in documents[:]:
        document.metadata = {"source": source_map[document.metadata["source"]]}
    return documents


class DocumentationDatasetLoader:
    """Loads the documentation dataset
    Usage:
    ```
    loader = DocumentationDatasetLoader()
    documents = loader.load()
    # save to disk
    loader.save_to_disk(path)
    # load from disk
    loader.load_from_disk(path)
    ```
    """

    def __init__(
        self,
        documentation_dir: str = "docodile",
        notebooks_dir: str = "examples/colabs/",
        code_dir: str = "examples/examples/",
        wandb_code_dir: str = "wandb",
        extra_data_dir: str = "extra_data",
        chunk_size: int = 768,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
    ):
        """
        :param documentation_dir: The directory containing the documentation from wandb/docodile
        :param notebooks_dir: The directory containing the wandb/examples/colab notebooks
        :param code_dir: The directory containing the wandb/examples/examples code
        :param extra_data_dir: The directory containing extra data to load
        :param chunk_size: The size of the chunks to split the text into using the `TokenTextSplitter`
        :param chunk_overlap: The amount of overlap between chunks of text using the `TokenTextSplitter`
        :param encoding_name: The name of the encoding to use when splitting the text using the `TokenTextSplitter`
        """
        self.documentation_dir = documentation_dir
        self.notebooks_dir = notebooks_dir
        self.code_dir = code_dir
        self.wandb_code_dir = wandb_code_dir
        self.extra_data_dir = extra_data_dir
        self.documents = []
        self.md_text_splitter = MarkdownTextSplitter()
        self.code_text_splitter = PythonCodeTextSplitter()
        self.token_splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            allowed_special={"<|endoftext|>"},
        )

    def load_documentation_documents(self, docs_dir: str) -> List[Document]:
        """
        Loads the documentation documents from the wandb/docodile repository
        :param docs_dir: The directory containing the documentation from wandb/docodile
        :return: A list of `Document` objects
        """
        document_files = load_documentation_paths(docs_dir=docs_dir)
        documents = []
        for f_name in tqdm(document_files, desc="Loading documentation"):
            try:
                documents.extend(UnstructuredMarkdownLoader(f_name).load())
            except:
                logger.warning(f"Failed to load documentation {f_name}")
        documents = map_source(documents, document_files)
        document_sections = self.md_text_splitter.split_documents(documents)
        document_sections = self.token_splitter.split_documents(document_sections)

        return document_sections

    def load_notebook_documents(
        self,
        notebook_dir: str,
        include_outputs: bool = True,
        max_output_length: int = 20,
        remove_newline: bool = True,
    ) -> List[Document]:
        """
        Loads the notebooks from the wandb/examples repository
        :param notebook_dir: The directory containing the wandb/examples/colab notebooks
        :param include_outputs: Whether to include the outputs of the notebook
        :param max_output_length: The maximum length of the output to include
        :param remove_newline: Whether to remove newlines from the output
        :return: A list of `Document` objects
        """
        notebook_files = load_notebook_paths(notebook_dir)
        notebooks = []
        for f_name in tqdm(notebook_files, desc="Loading notebooks"):
            try:
                notebooks.extend(
                    NotebookLoader(
                        f_name,
                        include_outputs=include_outputs,
                        max_output_length=max_output_length,
                        remove_newline=remove_newline,
                    ).load()
                )
            except:
                logger.warning(f"Failed to load notebook {f_name}")
        notebooks = map_source(notebooks, notebook_files)
        notebook_sections = self.code_text_splitter.split_documents(notebooks)
        notebook_sections = self.token_splitter.split_documents(notebook_sections)
        return notebook_sections

    def load_code_documents(self, code_dir: str, examples=True) -> List[Document]:
        """
        Loads the code documents from the wandb/examples repository
        :param code_dir: The directory containing the wandb/examples/examples code
        :return: A list of `Document` objects
        """
        code_files = load_code_paths(code_dir=code_dir, examples=examples)
        codes = []
        for f_name in tqdm(code_files, desc="Loading code"):
            try:
                contents = open(f_name, "r").read()
                codes.append(
                    Document(page_content=contents, metadata={"source": f_name})
                )
            except:
                logger.warning(f"Failed to load code {f_name}")

        codes = map_source(codes, code_files)
        code_sections = self.code_text_splitter.split_documents(codes)
        code_sections = self.token_splitter.split_documents(code_sections)
        return code_sections

    def load_extra_documents(self, extra_data_dir: str) -> List[Document]:
        extra_data = []
        for f_name in pathlib.Path(extra_data_dir).glob("*.csv"):
            extra_data.extend(load_csv_data(str(f_name)))

        documents = [
            Document(page_content=doc["reference"], metadata={"source": doc["source"]})
            for doc in tqdm(extra_data, desc="loading extra data")
        ]
        document_sections = self.token_splitter.split_documents(documents)
        return document_sections

    def load(self) -> List[Document]:
        """
        Loads the documentation, notebooks and code documents
        :return: A list of `Document` objects
        """
        self.documents = []
        if self.documentation_dir and os.path.exists(self.documentation_dir):
            self.documents.extend(
                self.load_documentation_documents(docs_dir=self.documentation_dir)
            )
        else:
            logger.warning(
                f"Documentation directory {self.documentation_dir} does not exist. Not loading documentation."
            )
        if self.notebooks_dir and os.path.exists(self.notebooks_dir):
            self.documents.extend(
                self.load_notebook_documents(notebook_dir=self.notebooks_dir)
            )
        else:
            logger.warning(
                f"Notebooks directory {self.notebooks_dir} does not exist. Not loading notebooks."
            )
        if self.code_dir and os.path.exists(self.code_dir):
            self.documents.extend(self.load_code_documents(code_dir=self.code_dir))
        else:
            logger.warning(
                f"Code directory {self.code_dir} does not exist. Not loading code."
            )
        if self.wandb_code_dir and os.path.exists(self.wandb_code_dir + "/wandb"):
            self.documents.extend(
                self.load_code_documents(code_dir=self.wandb_code_dir, examples=False)
            )
        else:
            logger.warning(
                f"Code directory {self.wandb_code_dir} does not exist. Not loading code."
            )
        if self.extra_data_dir and os.path.exists(self.extra_data_dir):
            self.documents.extend(self.load_extra_documents(self.extra_data_dir))
        else:
            logger.warning(
                f"Extra data directory {self.extra_data_dir} does not exist. Not loading extra data."
            )
        return self.documents

    def save_to_disk(self, path: str) -> None:
        """
        Saves the documents to disk as a jsonl file
        :param path: The path to save the documents to
        """
        with open(path, "w") as f:
            for document in self.documents:
                line = json.dumps(
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    }
                )
                f.write(line + "\n")

    @classmethod
    def load_from_disk(cls, path: str) -> "DocumentationDatasetLoader":
        """
        Loads the jsonl documents from disk into a `DocumentationDatasetLoader`
        :param path: The path to the jsonl file containing the documents
        :return: A `DocumentationDatasetLoader` object
        """
        loader = cls()
        with open(path, "r") as f:
            for line in f:
                document = json.loads(line)
                loader.documents.append(Document(**document))
        return loader


class DocumentStore:
    """
    A class for storing and retrieving documents using FAISS and OpenAI embeddings
    """

    base_embeddings = OpenAIEmbeddings()

    def __init__(
        self,
        documents: List[Document],
        use_hyde: bool = True,
        hyde_prompt: Optional[Union[ChatPromptTemplate, str]] = None,
        temperature: float = 0.7,
    ):
        """
        :param documents: List of documents to store in the document store
        :param use_hyde: Whether to use the hypothetical document embeddings when embedding documents
        :param hyde_prompt: The prompt to use for the hypothetical document embeddings
        :param temperature: The temperature to use for the hypothetical document embeddings
        """
        self.documents = documents
        self.use_hyde = use_hyde
        self.hyde_prompt = hyde_prompt
        self._embeddings = None
        self._faiss_store = None
        self.temperature = temperature

    def embeddings(self) -> Union[Chain, Embeddings]:
        """
        Returns the embeddings to use for the document store
        :return:
        """
        if self._embeddings is None:
            if self.use_hyde:
                if isinstance(self.hyde_prompt, ChatPromptTemplate):
                    prompt = self.hyde_prompt
                elif isinstance(self.hyde_prompt, str) and os.path.isfile(
                    self.hyde_prompt
                ):
                    prompt = load_hyde_prompt(self.hyde_prompt)
                else:
                    prompt = load_hyde_prompt()
                self._embeddings = HypotheticalDocumentEmbedder(
                    llm_chain=LLMChain(
                        llm=ChatOpenAI(temperature=self.temperature), prompt=prompt
                    ),
                    base_embeddings=self.base_embeddings,
                )
            else:
                self._embeddings = self.base_embeddings
        return self._embeddings

    def create_faiss_index(
        self,
    ) -> FAISS:
        """
        Creates a FAISS index from documents
        :return: A `FAISS` object
        """

        self._faiss_store = FAISS.from_documents(self.documents, self.embeddings())
        return self._faiss_store

    @property
    def faiss_index(
        self,
    ) -> FAISS:
        """
        Returns the FAISS index
        :return: A `FAISS` object
        """
        if self._faiss_store is None:
            self.create_faiss_index()
        return self._faiss_store

    def save_to_disk(self, path: str) -> None:
        """
        Saves the FAISS index to disk
        :param path: The directory to save the FAISS index to
        """
        self.faiss_index.save_local(path)

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        use_hyde: bool = True,
        hyde_prompt: Optional[Union[ChatPromptTemplate, str]] = None,
        temperature: float = 0.7,
    ) -> "DocumentStore":
        """
        Loads the `DocumentStore` from disk
        :param path: The directory the FAISS index
        :param use_hyde: Whether to use the hypothetical document embeddings when embedding documents
        :param hyde_prompt: The prompt to use for the hypothetical document embeddings
        :param temperature: The temperature to use for the hypothetical document embeddings
        :return: A `DocumentStore` object
        """
        cls.use_hyde = use_hyde
        cls.hyde_prompt = hyde_prompt
        cls.temperature = temperature
        cls._embeddings = None
        cls._faiss_store = FAISS.load_local(path, cls.embeddings(cls))
        obj = cls(
            list(cls._faiss_store.docstore._dict.values()),
            cls.use_hyde,
            cls.hyde_prompt,
        )
        obj._faiss_store = cls._faiss_store
        obj._embeddings = cls._embeddings
        return obj


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        required=True,
        help="The directory containing the wandb documentation",
    )
    parser.add_argument(
        "--notebooks_dir",
        type=str,
        help="The directory containing the colab notebooks from the wandb/examples repo",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        help="The directory containing the examples code from the wandb/examples repo",
    )
    parser.add_argument(
        "--wandb_code_dir",
        type=str,
        help="The directory containing the wandb sdk code from the wandb/examples repo",
    )
    parser.add_argument(
        "--extra_data_dir",
        type=str,
        help="The directory containing the extra data to add to the dataset",
    )
    parser.add_argument(
        "--documents_file",
        type=str,
        default="data/documents.jsonl",
        help="The path to save or load the documents to/from",
    )
    parser.add_argument(
        "--faiss_index",
        type=str,
        default="data/faiss_index",
        help="The directory to save or load the faiss index to/from",
    )
    parser.add_argument(
        "--hyde_prompt",
        type=str,
        default=None,
        help="The path to the hyde prompt to use",
    )
    parser.add_argument(
        "--use_hyde",
        action="store_true",
        help="Whether to use the hypothetical document embeddings",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="The temperature to use for the hypothetical document embeddings",
    )
    parser.add_argument(
        "--wandb_project",
        default="wandb_docs_bot",
        type=str,
        help="The wandb project to use for storing artifacts",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)

    if not os.path.isfile(args.documents_file):
        loader = DocumentationDatasetLoader(
            documentation_dir=args.docs_dir,
            notebooks_dir=args.notebooks_dir,
            code_dir=args.code_dir,
            wandb_code_dir=args.wandb_code_dir,
            extra_data_dir=args.extra_data_dir,
        )
        documents = loader.load()
        loader.save_to_disk(args.documents_file)
    else:
        loader = DocumentationDatasetLoader.load_from_disk(args.documents_file)
        documents = loader.documents

    documents_artifact = wandb.Artifact("docs_dataset", type="dataset")
    documents_artifact.add_file(args.documents_file)
    run.log_artifact(documents_artifact)
    if not os.path.isdir(args.faiss_index):
        document_store = DocumentStore(
            documents=documents,
            use_hyde=args.use_hyde,
            hyde_prompt=args.hyde_prompt,
            temperature=args.temperature,
        )
        document_store.save_to_disk(args.faiss_index)
    else:
        document_store = DocumentStore.load_from_disk(
            args.faiss_index,
            use_hyde=args.use_hyde,
            hyde_prompt=args.hyde_prompt,
            temperature=args.temperature,
        )
    faiss_index_artifact = wandb.Artifact("faiss_store", type="search_index")
    faiss_index_artifact.add_dir(args.faiss_index)
    run.log_artifact(faiss_index_artifact)

    if args.hyde_prompt is not None and os.path.isfile(args.hyde_prompt):
        hyde_prompt_artifact = wandb.Artifact("hyde_prompt", type="prompt")
        hyde_prompt_artifact.add_file(args.hyde_prompt)
        run.log_artifact(hyde_prompt_artifact)

    run.finish()


if __name__ == "__main__":
    main()
