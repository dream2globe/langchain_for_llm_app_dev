import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loguru import logger

if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
