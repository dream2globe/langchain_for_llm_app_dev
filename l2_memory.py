import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)
from loguru import logger

from l2_long_string import schedule


def buffer_memory_sample(llm: ChatOpenAI) -> None:
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.predict(input="Hi, my name is Andrew")
    conversation.predict(input="What is 1+1?")
    conversation.predict(input="What is my name?")
    logger.debug(memory.buffer)  # save the conversation until now
    logger.debug(memory.load_memory_variables({}))
    pass


def buffer_win_memory_sample(llm: ChatOpenAI) -> None:
    memory = ConversationBufferWindowMemory(
        k=1
    )  # only memorize the latest conversation
    memory.save_context({"input": "Hi"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    logger.debug(memory.load_memory_variables({}))
    pass


def buffer_win_memory_sample(llm: ChatOpenAI) -> None:
    memory = ConversationBufferWindowMemory(
        k=1
    )  # only memorize the latest conversation
    memory.save_context({"input": "Hi"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    logger.debug(memory.load_memory_variables({}))
    pass


def token_buffer_memory_sample(llm: ChatOpenAI) -> None:
    memory = ConversationTokenBufferMemory(
        llm=llm, max_token_limit=20
    )  # the way to count torkens of each model is different, so model param is requested.
    memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
    memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})
    logger.debug(memory.load_memory_variables({}))


def summary_memory_sample(llm: ChatOpenAI) -> None:
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    memory.save_context(
        {"input": "What is on the schedule today?"}, {"output": f"{schedule}"}
    )
    logger.debug(memory.load_memory_variables({}))
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.predict(input="What would be a good demo to show?")
    logger.debug(memory.load_memory_variables({}))


if __name__ == "__main__":
    ## initialize
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", verbose=True)

    ## samples
    # buffer_memory_sample(llm)
    # buffer_win_memory_sample(llm)
    # token_buffer_memory_sample(llm)
    summary_memory_sample(llm)


"""
Memory Types

ConversationBufferMemory
    This memory allows for storing of messages and then extracts the messages
    in a variable.

ConversationBufferWindowsMemory
    This memory keeps a list of the interactions of the conversation over time.
    It only uses the last K interactions.

ConversationTokernBufferMemory
    This memory keeps a buffer of recent interactions in memory, and uses token 
    length rather than number of interactions to dettemine when to flush
    interations.
    
ConversationSummaryMemory
    This memory creates a summary of the conversation over time.
    
Vector data memory
    Stores text(frome conversation or elsewhere) in a vector database and
    ertrieves the most relevant blocks of text.
    
Entity memories
    Using an LLM, it remembers details about specific entities
"""
