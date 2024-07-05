import os
from dotenv import load_dotenv, find_dotenv

def get_tavily_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['TAVILY_API_KEY']
