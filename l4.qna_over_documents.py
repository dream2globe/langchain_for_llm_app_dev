import os

import openai
from dotenv import find_dotenv, load_dotenv

# from IPython.display import Markdown, display
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from loguru import logger


def make_lst_simply(file_nm, query):
    loader = CSVLoader(file_path=file_nm)
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    response = index.query(query)
    logger.debug(response)


def make_lst_step_by_step(file_nm, query, llm_model):
    loader = CSVLoader(file_path=file_nm)
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    docs = db.similarity_search(query)
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    qdocs = "".join([docs[i].page_content for i in range(len(docs))])
    response = llm.call_as_llm(
        f"{qdocs} Question: Please list all your shirts with sun protection in a table in markdown and summarize each one."
    )
    # qa_stuff = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=retriever, verbose=True
    # )
    # response = qa_stuff.run(query)


if __name__ == "__main__":
    # init
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    # sample run
    make_lst_simply(
        file_nm="OutdoorClothingCatalog_1000.csv",
        query="Please list all your shirts with sun protection in a table in markdown and summarize each one.",
    )
    make_lst_step_by_step(
        file_nm="OutdoorClothingCatalog_1000.csv",
        query="Please list all your shirts with sun protection in a table in markdown and summarize each one.",
        llm_model=llm,
    )

"""
1. Stuff method
    - 모든 문장의 내용을 한 문장으로 연결하여 한 번에 처리하는 가장 간단한 방식
2. Map_reduce
    - 언어모델을 각 문장에 호출. 동시적 처리 가능  
3. Refine
    - 이전 결과를 연결해서 다음 질문으로 진행.
4. Map_rerank
    - 언어모델을 각 문장에 호출하여 각 문장에 점수를 부여함(예로 유사도)
        가장 높은 점수의 문장을 선정
"""
