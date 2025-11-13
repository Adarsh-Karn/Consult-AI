from llm_file import llm
from retriever_setup import prepare_retriever, learning_retriever, case_prep_retriever
from prepare_chain import load_prepare_chain
from learning_chain import load_learning_chain
from case_prep_chain import load_case_prep_chain
from case_example import load_case_examples_chain


def test_prepare():
    print("=== Testing prepare_chain ===")
    chain = load_prepare_chain(llm, prepare_retriever)
    ans = chain.invoke(
        {"question": "How should I approach a profitability case?"},
        config={"configurable": {"session_id": "test1"}}
    )
    print(ans)
    print()


def test_learning():
    print("=== Testing learning_chain ===")
    chain = load_learning_chain(llm, learning_retriever)
    ans = chain.invoke(
        {"question": "Explain MECE framework with examples"},
        config={"configurable":{"session_id": "test2"}}
    )
    print(ans)
    print()


def test_case_prep():
    print("=== Testing case_prep_chain ===")
    chain = load_case_prep_chain()
    ans = chain.invoke(
        {"input": "Start the Market Entry case"},
        config={"configurable":{"session_id": "test3"}}
    )
    print(ans)
    print()


def test_case_examples():
    print("=== Testing case_examples ===")
    chain = load_case_examples_chain()
    ans = chain.invoke(
        {"question": "Give me examples of growth strategy cases"},
        config={"configurable":{"session_id": "test4"}}
    )
    print(ans)
    print()


if __name__ == "__main__":
    test_prepare()
    test_learning()
    test_case_prep()
    test_case_examples()
