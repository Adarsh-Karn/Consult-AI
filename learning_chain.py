# learning_chain.py

from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from retriever_setup import learning_retriever
from llm_file import llm


def load_learning_chain(llm, learning_retriever):
    # Prompt used when combining retrieved documents
    combine_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a consulting tutor helping a candidate understand core consulting concepts.

Use the following context to answer the user's question.

Context:
{context}

Question: {question}

Answer:
"""
    )

    # ⭐ LCEL Chain: Retriever → Prompt → LLM → Response
    # Fixed: Properly pass question to retriever and combine docs
    def combine_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": itemgetter("question") | learning_retriever | combine_docs,
            "question": itemgetter("question")
        }
        | combine_prompt
        | llm
        | StrOutputParser() # type: ignore
    )

    # ⭐ Store chat history per session
    store = {}

    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # ⭐ Wrap chain with automatic chat-memory (replacement for ConversationBufferMemory)
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return conversational_chain


print("SUCCESS")

__all__ = ["load_learning_chain"]
