import datetime
import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from loguru import logger


def get_completion(prompt, model) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message["content"]


def call_api_directly(prompt: str) -> str:
    response = get_completion(prompt=prompt)
    return response


def chat_api_using_langchain(
    model, template: str, input_variables: dict[str, str]
) -> str:
    # prompt
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format_messages(**input_variables)
    logger.debug(prompt_template.messages[0].prompt)
    logger.debug(prompt_template.messages[0].prompt.input_variables)

    # model
    chat = ChatOpenAI(temperature=0.0, model=model)

    # call the llm
    customer_response = chat(prompt)
    return customer_response.content


if __name__ == "__main__":
    # initialize
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]
    current_date = datetime.datetime.now().date()
    target_date = datetime.date(2024, 6, 12)

    if current_date > target_date:
        llm_model = "gpt-3.5-turbo"
    else:
        llm_model = "gpt-3.5-turbo-0301"

    # TASK 1: Translate a text
    translate_template = """Translate the text \
    that is delimited by triple backticks \
    into a style that is {style}. \
    text: ```{text}```
    """
    customer_style = """American English \
    in a calm and respectful tone
    """
    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse, \
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """
    # input_variables = {"style": customer_style, "text": customer_email}
    # answer = chat_api_using_langchain(
    #     model=llm_model, template=translate_template, input_variables=input_variables
    # )
    # logger.debug(answer)

    # TASK 2: Review summarization
    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """
    review_template = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """
    # input_variables = {"text": customer_review}
    # answer = chat_api_using_langchain(
    #     model=llm_model, template=review_template, input_variables=input_variables
    # )
    # logger.debug(answer)

    # TASK 3: Review summarization using a parser
    gift_schema = ResponseSchema(
        name="gift",
        description="Was the item purchased\
as a gift for someone else? \
Answer True if yes,\
False if not or unknown.",
    )
    delivery_days_schema = ResponseSchema(
        name="delivery_days",
        description="How many days\
did it take for the product\
to arrive? If this \
information is not found,\
output -1.",
    )
    price_value_schema = ResponseSchema(
        name="price_value",
        description="Extract any\
sentences about the value or \
price, and output them as a \
comma separated Python list.",
    )
    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    review_template_2 = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product\
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    text: {text}

    {format_instructions}
    """
    input_variables = {
        "text": customer_review,
        "format_instructions": format_instructions,
    }
    answer = chat_api_using_langchain(
        model=llm_model, template=review_template_2, input_variables=input_variables
    )
    logger.debug(answer)
    output_dict = output_parser.parse(answer)
    logger.debug(output_dict.get("delivery_days"))
