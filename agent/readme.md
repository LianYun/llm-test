
# LangGraph

## Summary

+ Cyclic Graphs: Extension of LangChain, support graphs, Single or Multi-agent flows
+ Persistence:
+ Human-in-the-loop: workflows

## Components

+ Nodes: Agents or functions
+ Edges: connect nodes
+ Conditional Edges: decisions

## AgentState

Can be accessible to all parts of the graph, local to the graph, can be stored in a persistence layer.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class ComplexAgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
```
