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


def router_chain():
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""

    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:PromptTemplate
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

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

    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

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

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
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
    # simple_sequential_chain("Queen Size Sheet Set")
    # logger.debug(sequential_chain(df.Review[5]))

    chain = router_chain()
    chain.run("what is black body radiation?")
    chain.run("what is 2 + 2")
    chain.run("Why does every cell in our body contain DNA?")
