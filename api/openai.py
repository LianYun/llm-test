import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv, find_dotenv

def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=get_openai_key())

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    return get_completion_from_messages(messages, model)

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def get_completion_and_token_count(messages,
                                   model="gpt-3.5-turbo",
                                   temperature=0,
                                   max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    token_dict = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return content, token_dict


def print_moderation_response(input):
    response = client.moderations.create(input=input)
    moderation_output = pd.DataFrame(response.results[0].category_scores)
    moderation_output = moderation_output.join(pd.DataFrame(response.results[0].categories).set_index(0), on=0, rsuffix='categories')
    print(moderation_output)    
