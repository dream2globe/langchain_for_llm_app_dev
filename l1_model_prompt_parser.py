import datetime
import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from loguru import logger

from l1_prompt import (
    customer_email,
    customer_review,
    customer_style,
    review_template,
    review_template_2,
    translate_template,
)


def get_completion(prompt) -> str:
    ## Select the llm model
    current_date = datetime.datetime.now().date()
    target_date = datetime.date(2024, 6, 12)
    if current_date > target_date:
        llm_model = "gpt-3.5-turbo"
    else:
        llm_model = "gpt-3.5-turbo-0301"

    ## Run the llm model
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=llm_model, messages=messages, temperature=0
    )
    return response.choices[0].message["content"]


def call_api_directly(prompt: str) -> str:
    response = get_completion(prompt=prompt)
    return response


def chat_api_using_langchain(template: str, input_variables: dict[str, str]) -> str:
    ## prompt
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format_messages(**input_variables)
    logger.debug(prompt_template.messages[0].prompt)
    logger.debug(prompt_template.messages[0].prompt.input_variables)

    ## model
    current_date = datetime.datetime.now().date()
    target_date = datetime.date(2024, 6, 12)
    if current_date > target_date:
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-3.5-turbo-0301"
    chat = ChatOpenAI(temperature=0.0, model=model)

    ## call the llm
    customer_response = chat(prompt)
    return customer_response.content


def translate_style() -> str:
    return chat_api_using_langchain(
        template=translate_template,
        input_variables={"style": customer_style, "text": customer_email},
    )


def translate_review() -> str:
    return chat_api_using_langchain(
        template=review_template, input_variables={"text": customer_review}
    )


def translate_review_output_parser() -> dict:
    gift_schema = ResponseSchema(
        name="gift",
        description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.",
    )
    delivery_days_schema = ResponseSchema(
        name="delivery_days",
        description="How many days did it take for the product to arrive? If this information is not found, output -1.",
    )
    price_value_schema = ResponseSchema(
        name="price_value",
        description="Extract any sentences about the value or price, and output them as a comma separated Python list.",
    )
    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser.parse(
        chat_api_using_langchain(
            template=review_template_2,
            input_variables={
                "text": customer_review,
                "format_instructions": output_parser.get_format_instructions(),
            },
        )
    )


if __name__ == "__main__":
    # initialize
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # logger.debug(get_completion("What is 1+1?"))
    # logger.debug(translate_style())
    # logger.debug(translate_review())
    logger.debug(translate_review_output_parser())
