from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def load_prepare_chain(llm, prepare_retriever):

    # --- Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are a consulting coach. Use ONLY the retrieved context to answer the user's question.
        If the context does not contain the answer, say the info is not available.
        """),
        MessagesPlaceholder("chat_history"),
        ("system", "Retrieved context:\n{context}"),
        ("human", "{question}")
    ])

    # --- Combine retrieved docs ---
    combine_docs = RunnableLambda(lambda docs:
        "\n\n".join(d.page_content for d in docs)
    )

    # --- Retrieval pipeline ---
    retrieval_pipeline = {
        "context": prepare_retriever | combine_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    # --- Memory ---
    def get_memory(session_id: str):
        return InMemoryChatMessageHistory()

    return RunnableWithMessageHistory(
        retrieval_pipeline,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
