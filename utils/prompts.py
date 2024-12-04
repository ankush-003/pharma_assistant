from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

class MemoryKV(BaseModel):
    """
    Store the key-value pair in memory, covert all values to string
    """
    value: dict[str, str] = Field(..., title="value", description="The key-value pair of the topic to store in memory")

class Grade(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


# Prompts
memory_update_prompt = PromptTemplate.from_template("""
    You are a precise information extraction system. Your task is to analyze conversations and extract key information into a structured format. Only extract information that is explicitly stated in the conversation.
    the conversation: {conversation} 
    """)

grade_document_prompt = PromptTemplate.from_template("""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """)