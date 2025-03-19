from typing import Tuple, Dict
import dotenv
import os
from dotenv import load_dotenv
import requests
import json
import streamlit as st
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
EXCHANGERATE_API_KEY = os.getenv('EXCHANGERATE_API_KEY')
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

# Initialize the OpenAI client
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def get_exchange_rate(base: str, target: str, amount: float) -> Tuple:
    """Return a tuple of (base, target, amount, conversion_result (2 decimal places))"""
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base}/{target}/{amount}"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("Error fetching exchange rate data.")
        return None

    data = response.json()
    try:
        conversion_result = data["conversion_result"]
    except KeyError:
        st.error("Invalid response from the exchange rate API.")
        return None

    return (base, target, amount, f'{conversion_result:.2f}')

def call_llm(textbox_input: str) -> Dict:
    """Make a call to the LLM with the textbox_input as the prompt.
       The output from the LLM should be a JSON (dict) with the base, amount and target"""
    
    tools = [{
                "type": "function",
                "function": {
                    "name": "exhange_rate_function",
                    "description": "Convert a given amount of money from one currency to another. Each currency will be represented as a 3 letter code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "base": {
                                "type": "string",
                                "description": "The base or original currency."
                            },
                            "target": {
                                "type": "string",
                                "description": "The targert or converted currency."
                            },
                            "amount": {
                                "type": "string",
                                "description": "The amount of money to convert from the base currency."
                            }
                        },
                        "required": [
                            "base",
                            "target",
                            "amount"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }]

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": textbox_input,
                }
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name,
            tools=tools
        )
        return response#.choices[0].message.content
    except Exception as e:
        st.error(f"Exception {e} for input: {textbox_input}")
        return None

def run_pipeline(user_input):
    """Based on textbox_input, determine if you need to use the tools (function calling) for the LLM.
    Call get_exchange_rate(...) if necessary"""

    response = call_llm(user_input)
    #st.write(response)
    if response.choices[0].finish_reason == "tool_calls":
        response_arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        base = response_arguments["base"]
        target = response_arguments["target"]
        amount = response_arguments["amount"]
        _, _, _, conversion_result = get_exchange_rate(base, target, amount)
        st.write(f'{base} {amount} is {target} {conversion_result}')
    elif response.choices[0].finish_reason == "stop":
        # Update this
        st.write(f"(Function calling not used) and {response.choices[0].message.content}")
    else:
        st.write("NotImplemented")
# Title of the application
st.title("Multilingual Money Changer")

# Text box for user input
user_input = st.text_input("Enter your text:")

# Submit button
if st.button("Submit"):
    if user_input:
        run_pipeline(user_input)
    else:
        st.error("Please enter some text.")