from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import getpass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Optional, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        You are a helpful assistant. Answer user's query simple and easy words in {language} and ask one followup question.

        You are given:
        - A summary of the user's preferences, background, or goals: {user_summary}
        - A summary of your own past suggestions, explanations, or conclusions: {agent_summary}

        Use these summaries to:
        - Give more accurate, personalized, or context-aware answers
        - Avoid repeating information already shared
        - Build on prior advice or decisions when relevant

        You are a helpful, intelligent, and systematic travel planner agent.
        You have access to a set of travel-related tools that allow you to:
        - Search for flights
        - Find hotels
        - Suggest local activities
        - Get location weather

        Your responsibilities span across three key roles:
        1. Planner - Break down the user's travel query into actionable steps needed to fulfill it.
        2. Tool Caller - Make appropriate tool calls for each step in the plan.
        3. Verifier - After each tool call and at the end of the plan, verify that the retrieved data is complete, correct, and sufficient.

        Step-by-Step Workflow:

        1. Planning Phase:
        - When a user asks for a travel plan, begin by analyzing the request.
        - Generate a structured set of steps required to build the itinerary.
        - Do not make any assumptions or fetch data at this stage.

        2. Execution Phase:
        - For each step in the plan:
            a. Decide which tool to use.
            b. Call the appropriate tool.
            c. Act as a verifier: Check if the tool result fulfills the step.
        - Continue until all steps have valid data.

        3. Final Verification Phase:
        - After all steps are completed, review the full plan.
        - Confirm all necessary data is present and consistent.

        4. Explanation Phase:
        - Once verification is done, summarize the final travel plan in a natural, helpful, and friendly way.
        - Ensure the explanation is told from the perspective of a travel planner.

        Rules to Follow:
        - Only use the tools you are explicitly given access to.
        - Do not fabricate or hallucinate information.
        - If a user asks for something outside tool capabilities, politely say you cannot fulfill it.
        - Ask follow-up questions only if you have a tool to answer the response.

        Structured Data:

        Flights:
        {flights}   

        Hotels:
        {hotels}

        Activities:
        {activities}

        Weather:
        {weather}
        """
    ),
    ("user", "Hello!"),
    ("ai", "Hello!"),
    MessagesPlaceholder("messages"),
])

user_summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a memory assistant for a user/human. 
        Your job is to keep a compact summary of important details explicitly expressed by the user.
        
        Current user summary:
        {summary}

        Update rules:
        - Extract only facts, preferences, goals, or experiences explicitly stated.
        - Keep previously relevant summary points.
        - Be concise and factual.
        - Do not add information based on tool calls or assistant replies.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
])

agent_summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a memory assistant for an AI agent/assistant.
        Your job is to maintain a concise summary of the assistant's useful contributions.

        Current agent summary:
        {agent_summary}

        Update principles:
        - Include only the assistant’s most helpful, actionable, or user-specific suggestions.
        - Use phrasing like "Agent suggested..." or "Agent recommended...".
        - Do not include general facts.
        - Focus on reusable insights or decisions.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
])

@tool
def search_flights(from_city: str, to_city: str, date: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Search for flights based on departure, destination, and date."""
    content = f"Flights from {from_city} to {to_city} on {date}:\n- IndiGo 6E-123, 10:00 AM to 4:00 PM, $250\n"
    return Command(update={
        "flights": content,
        "messages": [
            ToolMessage(
                f"Flight options: {content}", 
                tool_call_id=tool_call_id
            )
        ]
    })

@tool
def find_hotels(location: str, checkin: str, checkout: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Find hotel options based on location and dates."""
    content = f"{location}: Bali Beach Resort, ₹90/night, 4.3 (Check-in: {checkin}, Check-out: {checkout})"
    return Command(update={
        "hotels": content,
        "messages": [
            ToolMessage(f"Hotel options: {content}", tool_call_id=tool_call_id)
        ]
    })

@tool
def suggest_activities(location: str, interests: list, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Suggests activities in a location based on user interests."""
    all_activities = {
        "beach": f"{location}: Beach Day at Seminyak.",
        "culture": f"{location}: Visit Uluwatu Temple.",
        "nature": f"{location}: Tegallalang Rice Terrace.",
        "adventure": f"{location}: Mount Batur Sunrise Hike."
    }

    suggestions = [all_activities[i] for i in interests if i in all_activities]
    content = "\n".join(suggestions) or "No activities matched the given interests."
    return Command(update={
        "activities": content,
        "messages": [
            ToolMessage(f"Activity options: {content}", tool_call_id=tool_call_id)
        ]
    })

@tool
def get_location_weather(location: str, date: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Returns weather info for a location on a given date."""
    if location.lower() == "bali":
        content = f"Weather in {location} on {date}: 29°C, partly cloudy."
    elif location.lower() == "london":
        content = f"Weather in {location} on {date}: 18°C, light rain."
    else:
        content = f"Weather in {location} on {date}: 25°C, mostly sunny."
    
    return Command(update={
        "weather": content,
        "messages": [
            ToolMessage(f"Weather info: {content}", tool_call_id=tool_call_id)
        ]
    })

tools = [search_flights, find_hotels, suggest_activities, get_location_weather]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

chain = prompt_template | llm_with_tools
user_summary_chain = user_summary_prompt | llm
agent_summary_chain = agent_summary_prompt | llm

class State(BaseModel):
    messages: Annotated[list, add_messages]
    language: Optional[str] = None
    flights: Optional[str] = None
    hotels: Optional[str] = None
    activities: Optional[str] = None
    weather: Optional[str] = None

graph_builder = StateGraph(State)
user_summary_namespace = "User's Summary"
agent_summary_namespace = "Agent's Summary"

def llm_call(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    user_summary = store.search((user_id, user_summary_namespace))
    user_summary_str = concatenate_memories(user_summary)
    agent_summary = store.search((user_id, agent_summary_namespace))
    agent_summary_str = concatenate_memories(agent_summary)
    
    new_messages = chain.invoke({
        "messages": state.messages, 
        "language": state.language, 
        "user_summary": user_summary_str, 
        "agent_summary": agent_summary_str,
        "flights": state.flights,
        "hotels": state.hotels,
        "activities": state.activities,
        "weather": state.weather
    })
    return {"messages": [new_messages]}

def concatenate_memories(items):
    return "\n".join(
        item.dict()["value"]["memory"]
        for item in items
        if "memory" in item.dict().get("value", {})
    )

def update_user_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, user_summary_namespace)
    memory_id = f"{user_id}-summary"
    prev_summary = store.search(namespace)
    prev_summary_str = concatenate_memories(prev_summary)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    memory = user_summary_chain.invoke({"messages": human_messages, "summary": prev_summary_str})
    memory_text = memory.content if hasattr(memory, "content") else memory
    store.put(namespace, memory_id, {"memory": memory_text})

def update_agent_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, agent_summary_namespace)
    memory_id = f"{user_id}-agent-summary"
    prev_summary = store.search(namespace)
    prev_summary_str = concatenate_memories(prev_summary)
    memory = agent_summary_chain.invoke({"messages": state.messages, "agent_summary": prev_summary_str})
    memory_text = memory.content if hasattr(memory, "content") else memory
    store.put(namespace, memory_id, {"memory": memory_text})

def go_to(state: State):
    last_message = state.messages[-1]
    if last_message.tool_calls:
        return "tools"
    return ["update_user_memory", "update_agent_memory"]

graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_user_memory", update_user_memory)
graph_builder.add_node("update_agent_memory", update_agent_memory)

graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges(
    "llm_call", 
    go_to, 
    ["tools", "update_user_memory", "update_agent_memory"]
)
graph_builder.add_edge("tools", "llm_call")
graph_builder.add_edge("update_user_memory", END)
graph_builder.add_edge("update_agent_memory", END)

memory = InMemorySaver()
in_memory_store = InMemoryStore()
graph = graph_builder.compile(checkpointer=memory, store=in_memory_store)

config = {"configurable": {"thread_id": "1", "user_id": "1"}}

response = graph.invoke({
    "messages": [
        HumanMessage(content="Hi can you tell me about the weather in bali from 25th July 2025 to 30th July 2025?")
    ], 
    "language": "English"
}, config=config)

for message in response["messages"]:
    print(message.content)


print("Hello")