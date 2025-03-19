from typing import Tuple, Dict
import dotenv
import os
import requests
import streamlit as st
import json
from dotenv import load_dotenv
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
            model=model_name
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Exception {e} for input: {textbox_input}")
        return None

def run_pipeline(user_input: str):
    """Based on user_input, call the necessary functions for the LLM and exchange rates."""
    # For this example, let's assume user_input is like: "Convert 100 USD to EUR"
    parts = user_input.split()
    if len(parts) == 5 and parts[0].lower() == 'convert':
        amount = float(parts[1])
        base_currency = parts[2].upper()
        target_currency = parts[4].upper()
        
        exchange_response = get_exchange_rate(base_currency, target_currency, amount)
        if exchange_response:
            st.write(f'{amount} {base_currency} is {exchange_response[3]} {target_currency}')
    else:
        llm_response = call_llm(user_input)
        st.write("Response from LLM:", llm_response)

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