import os

import openai
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from loguru import logger

from l3_prompt import *


def simple_sequential_chain(product: str) -> None:
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
    prompt_1 = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}?"
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)
    prompt_2 = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following company:{company_name}"
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)
    overall_simple_chain = SimpleSequentialChain(
        chains=[chain_1, chain_2], verbose=True
    )
    overall_simple_chain.run(product)


def regular_sequential_chain(review: str) -> None:
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to english:" "\n\n{Review}"
    )
    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
    )
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up respones to the following "
        "summary in the specified language: "
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )

    chain_1 = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")
    chain_2 = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")
    chain_3 = LLMChain(llm=llm, prompt=third_prompt, output_key="language")
    chain_4 = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")
    overall_chain = SequentialChain(
        chains=[chain_1, chain_2, chain_3, chain_4],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "followup_message"],
        verbose=True,
    )
    logger.debug(overall_chain(review))


def router_chain():
    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template,
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template,
        },
        {
            "name": "History",
            "description": "Good for answering history questions",
            "prompt_template": history_template,
        },
        {
            "name": "computer science",
            "description": "Good for answering computer science questions",
            "prompt_template": computerscience_template,
        },
    ]
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    router_template = multi_prompt_router_template.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )
    return chain


if __name__ == "__main__":
    # init
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]
    df = pd.read_csv("data.csv", sep="\t")

    # Run chains
    ## sequential chains
    simple_sequential_chain("Queen Size Sheet Set")
    regular_sequential_chain(review=df.Review[5])

    ## rounter chains
    chain = router_chain()
    logger.debug(chain.run("what is black body radiation?"))
    logger.debug(chain.run("what is 2 + 2"))
    logger.debug(chain.run("Why does every cell in our body contain DNA?"))
