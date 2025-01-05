import json
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
_ = load_dotenv()

    
# tavily: https://python.langchain.com/docs/integrations/tools/tavily_search/
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_images=True)
print(f"toolType= {type(tool)}, toolName= {tool.name}")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = json.dumps(self.tools[t['name']].invoke(t['args']))
            
            content = f"tool return content: {str(result)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=content))
        print("Back to the model!")
        return {'messages': results}
    
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

def get_o1_mini_model():
    model = ChatOpenAI(model="gpt-4o-mini")  #reduce inference cost
    return model

def get_deepseek_model():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL")
    model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url) 
    return model

def get_stepfun_model():
    api_key = os.getenv("STEPFUN_API_KEY")
    base_url = os.getenv("STEPFUN_BASE_URL")
    model = os.getenv("STEPFUN_MODEL")
    model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url) 
    return model

def get_zhipu_model():
    api_key = os.getenv("ZHIPU_API_KEY")
    base_url = os.getenv("ZHIPU_BASE_URL")
    model = os.getenv("ZHIPU_MODEL")
    model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url) 
    return model


def get_doubao_model():
    api_key = os.getenv("DOUBAO_API_KEY")
    base_url = os.getenv("DOUBAO_BASE_URL")
    model = os.getenv("DOUBAO_MODEL")
    model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url) 
    return model

abot = Agent(get_o1_mini_model(), [tool], system=prompt)

# from IPython.display import Image
# Image(abot.graph.get_graph().draw_png())

# messages = [HumanMessage(content="What is the weather in sf?")]
# result = abot.graph.invoke({"messages": messages})
 

messages = [
    HumanMessage(content="What is the weather in SF and LA?")
]
result = abot.graph.invoke({"messages": messages})
print(result)


# query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
# What is the GDP of that state? Answer each question." 
# messages = [HumanMessage(content=query)]

# # model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
# abot = Agent(model, [tool], system=prompt)
# result = abot.graph.invoke({"messages": messages})
