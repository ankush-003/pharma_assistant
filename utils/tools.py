from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

## Custom tools
@tool 
def web_search_tool(query: Annotated[str, "the query to search on the internet"]) -> str:
    """
    searches the query on DuckDuckGo and returns the content of the search results
    arguments:
        query: str: the query to search
    returns:
        str: the search results concatenated from multiple pages
    """
    search = DuckDuckGoSearchResults(output_format="list", handle_tool_error=True, num_results=2)
    results = search.invoke(query)
    links = [result["link"] for result in results]
    loader_multiple_pages = WebBaseLoader(links)
    docs = loader_multiple_pages.load()
    res = "\n".join([doc.page_content for doc in docs])
    return res

## Utility functions
def get_web_docs(query: Annotated[str, "the query to search on the internet"]) -> str:
    """
    searches the query on DuckDuckGo and returns the list of links
    arguments:
        query: str: the query to search
    returns:
        str: the search results concatenated from multiple pages
    """
    search = DuckDuckGoSearchResults(output_format="list", handle_tool_error=True, num_results=10)
    results = search.invoke(query)
    # results = [{"link": "https://www.linkedin.com/in/ankush003/"}, {"link": "https://www.researchgate.net/profile/Ankush-H-V-2"}, {"link":"https://github.com/ankush-003"}]
    links = [result["link"] for result in results]
    loader_multiple_pages = WebBaseLoader(links)
    docs = loader_multiple_pages.load()
    # res = "\n".join([doc.page_content for doc in docs])
    return docs


def get_retriever_tool(topic: str, docs, embedding_model, store):
    """
    Returns a retriever tool with the given topic, documents, embedding model and store.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=10
    )
    # docs_list = [item for sublist in docs for item in sublist]
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = store.from_documents(documents=doc_splits, embedding=embedding_model)
    return get_prebuilt_retriever_tool(topic, vectorstore)

def get_prebuilt_retriever_tool(topic: str, store):
    """
    Returns a retriever tool with the given topic and store.
    """
    return create_retriever_tool(store.as_retriever(), "doc_retriever", f"""
        Document store created from the web search results on the query: {topic}.
    """)

def embed_docs(docs, embedding_model, store, ids: list = None):
    """
    Returns an embedding tool with the given documents, embedding model and store.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=10
    )
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = store.from_documents(documents=doc_splits, embedding=embedding_model)
    return vectorstore