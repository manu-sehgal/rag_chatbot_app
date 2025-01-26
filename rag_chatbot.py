from huggingface_hub import login
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os
import streamlit as st


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RAGChatbot:
    def __init__(self, document_path, hf_token=None):
        # Allow token to be passed directly or from Streamlit secrets
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.error("Please provide a HuggingFace token")
            raise ValueError("HuggingFace token is required")

        # Login to HuggingFace
        login(token=hf_token)

        self.llm = HuggingFaceEndpoint(
            repo_id="google/gemma-2-2b-it",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        self.vector_store = InMemoryVectorStore(self.embeddings)
        self._load_and_index_document(document_path)
        self._create_rag_graph()

    def _load_and_index_document(self, document_path):
        loader = PyPDFLoader(file_path=document_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents=all_splits)

    def _create_rag_graph(self):
        prompt = hub.pull("rlm/rag-prompt")

        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke(
                {"question": state["question"], "context": docs_content}
            )
            response = self.llm.invoke(messages)
            return {"answer": response}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def invoke(self, input_dict):
        return self.graph.invoke(input_dict)


def initialize_rag_system(document_path, hf_token=None):
    return RAGChatbot(document_path, hf_token)
