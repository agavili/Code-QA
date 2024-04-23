from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

import os
import string
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


def load_repo(url):
    repo = Repo.clone_from(url, to_path='temp_repo/')
    loader = GenericLoader.from_filesystem(
        os.path.join(os.getcwd(), 'temp_repo/'),
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    return documents


def get_document_info(documents):
    script_names = []
    code = []
    for doc in documents:
        script_names.append(os.path.basename(doc.metadata["source"]))
        code.append(doc.page_content)
    return script_names, code


def upload_docs_db(documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200)
    texts = python_splitter.split_documents(documents)

    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    return retriever
