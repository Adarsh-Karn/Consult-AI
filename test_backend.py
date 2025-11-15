# test_backend.py
"""
This file tests all chains individually using proper LCEL invocation format
and correct config structure required by RunnableWithMessageHistory.
"""

from prepare_chain import load_prepare_chain
from learning_chain import load_learning_chain
from case_prep_chain import load_case_prep_chain
from case_example import load_case_examples_chain

from retriever_setup import (
    prepare_retriever,
    learning_retriever,
    case_prep_retriever
)

from llm_file import llm


# ---------------------- TEST PREPARE CHAIN ----------------------

def test_prepare():
    print("\n=== Testing prepare_chain ===")

    chain = load_prepare_chain(llm, prepare_retriever)

    ans = chain.invoke(
        {"question": "How should I approach a profitability case?"},
        config={"configurable": {"session_id": "test_prepare"}}
    )

    print("\n--- Output ---")
    print(ans)


# ---------------------- TEST LEARNING CHAIN ----------------------

def test_learning():
    print("\n=== Testing learning_chain ===")

    chain = load_learning_chain(llm, learning_retriever)

    ans = chain.invoke(
        {"question": "Explain the MECE principle like I'm learning it from scratch."},
        config={"configurable": {"session_id": "test_learning"}}
    )

    print("\n--- Output ---")
    print(ans)


# ---------------------- TEST CASE PREP CHAIN ----------------------

def test_case_prep():
    print("\n=== Testing case_prep_chain ===")

    chain = load_case_prep_chain()

    ans = chain.invoke(
        {"input": "Start a market entry case."},
        config={"configurable": {"session_id": "test_case_prep"}}
    )

    print("\n--- Output ---")
    print(ans)


# ---------------------- TEST CASE EXAMPLES CHAIN ----------------------

def test_case_examples():
    print("\n=== Testing case_examples_chain ===")

    chain = load_case_examples_chain()

    ans = chain.invoke(
        {"question": "Give me examples of profitability cases."},
        config={"configurable": {"session_id": "test_case_examples"}}
    )

    print("\n--- Output ---")
    print(ans)


# ---------------------- RUN ALL TESTS ----------------------

if __name__ == "__main__":
    test_prepare()
    test_learning()
    test_case_prep()
    test_case_examples()
