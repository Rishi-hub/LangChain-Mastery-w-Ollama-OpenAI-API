import asyncio
import aiohttp
import os

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv() # Load environment variables from .env file into os.environ

# Constants and Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set!")
OPENAI_API_KEY = SecretStr(OPENAI_API_KEY)
# SerpAPI API Key
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    raise RuntimeError("SERPAPI_API_KEY environment variable not set!")
SERPAPI_API_KEY = SecretStr(SERPAPI_API_KEY)

# LLM and Prompt Setup
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    streaming=True,
    api_key=OPENAI_API_KEY
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided back to you. When you have "
        "all the information you need, you MUST use the final_answer tool "
        "to provide a final answer to the user. Use tools to answer the "
        "user's CURRENT question, not previous questions."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# we use the article object for parsing serpapi results later
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )

# Tools definition
# note: we define all tools as async to simplify later code, but only the serpapi
# tool is actually async
@tool
async def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
async def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
async def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
async def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

@tool
async def divide(x: float, y: float) -> float:
    """Divide 'y' by 'x'."""
    if x == 0:
        raise ValueError("Cannot divide by zero.")
    return y / x

@tool
async def serpapi(query: str) -> list[Article]:
    """Use this tool to search the web."""
    params = {
        "api_key": SERPAPI_API_KEY.get_secret_value(),
        "engine": "google",
        "q": query,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://serpapi.com/search",
            params=params
        ) as response:
            results = await response.json()
    return [Article.from_serpapi_result(result) for result in results["organic_results"]]

@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final `answer` to the user.
    The `answer` should be in natural language as this will be provided
    to the user directly. The `tools_used` must include a list of tool
    names that were used within the `scratchpad`. You MUST use this tool
    to conclude the interaction.
    """
    return {"answer": answer, "tools_used": tools_used}

@tool
async def best_guess(guess: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide `guess`, your best educated response to 
    the user based on the information you currently have.
    The `guess` should be in natural language as this will be provided
    to the user directly. The `guess` should also include a confidence 
    level in said answer so the user has an idea of your level of certitude. 
    The `tools_used` must include a list of tool
    names that were used within the `scratchpad`. DO NOT use this tool unless
    specifically asked for best_guess in the prompt. Use this tool if 
    final_answer and its tool abservation are NOT present in agent `scratchpad`.
    """
    return {"answer": guess, "tools_used": tools_used}

tools = [add, subtract, multiply, exponentiate, divide, final_answer, serpapi, best_guess]
# note when we have sync tools we use tool.func, when async we use tool.coroutine
name2tool = {tool.name: tool.coroutine for tool in tools}

# Streaming Handler
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return
            if token_or_done:
                yield token_or_done
    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")

async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = await name2tool[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )

# Agent Executor
class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 3):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        # streaming function
        async def stream(query: str) -> list[AIMessage]:
            response = self.agent.with_config(
                callbacks=[streamer]
            )
            # we initialize the output dictionary that we will be populating with
            # our streamed output
            outputs = []
            # now we begin streaming
            async for token in response.astream({
                "input": query,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            }):
                tool_calls = token.additional_kwargs.get("tool_calls")
                if tool_calls:
                    # first check if we have a tool call id - this indicates a new tool
                    if tool_calls[0]["id"]:
                        outputs.append(token)
                    else:
                        outputs[-1] += token
                else:
                    pass
            return [
                AIMessage(
                    content=x.content,
                    tool_calls=x.tool_calls,
                    tool_call_id=x.tool_calls[0]["id"]
                ) for x in outputs
            ]

        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            tool_calls = await stream(query=input)
            # gather tool execution coroutines
            tool_obs = await asyncio.gather(
                *[execute_tool(tool_call) for tool_call in tool_calls]
            )
            
            # agent_scratchpad.extend(tool_calls)
            # agent_scratchpad.extend(tool_obs)
            """
            [
                AIMessage(tool_call_id="A")
                AIMessage(tool_call_id="B")
                ToolMessage(tool_call_id="A")
                ToolMessage(tool_call_id="B")
            ]
            """
            # Problem with above is that llm agent doesn't connect the AIMessage tool call to the ToolMessage tool observation 
            # despite having the same id returning nothing and freezing the agent. 
            # Instead, we need to ensure that corresponding tool calls and tool observations
            # are together on after another in the agent_scratchpad.
            
            # We need to append them in order, so we create a mapping of tool call id to tool observation
            # and iterate through the tool calls to append them in order
            # Note: this is a kind of a workaround, ideally the llm should handle this for us
            # but it doesn't currently do so, so we do it manually here.
            # Explore possible solutions to this

            # append tool calls and tool observations to the scratchpad in order
            id2tool_obs = {
                tool_call.tool_call_id: tool_obs
                for tool_call, tool_obs in zip(tool_calls, tool_obs)
            }
            for tool_call in tool_calls:
                agent_scratchpad.extend([
                    tool_call, # AIMessage
                    id2tool_obs[tool_call.tool_call_id] # ToolMessage
                ])
            
            count += 1
            # if the tool call is the final answer tool, we stop
            found_final_answer = False
            for tool_call in tool_calls:
                if tool_call.tool_calls[0]["name"] == "final_answer":
                    final_answer_call = tool_call.tool_calls[0]
                    final_answer = final_answer_call["args"]["answer"]
                    found_final_answer = True
                    break
            
            # Only break the loop if we found a final answer
            if found_final_answer:
                break
            
        # if we didn't find a final answer, we need to use the best_guess tool
        # to provide a best guess answer
        if not final_answer:
            # if we didn't find a final answer, we return the best guess tool call
            # Use stream to generate an agent LLM call and stream it to the user
            tool_calls = await stream("I'm unable to provide a final answer and need to give a `best_guess` response:\n" + input)
            tool_obs = await asyncio.gather(
                *[execute_tool(tool_call) for tool_call in tool_calls]
            )
            # Append the tool calls and observations to the scratchpad
            for tool_call, tool_obs in zip(tool_calls, tool_obs):
                agent_scratchpad.extend([
                    tool_call,  # AIMessage
                    tool_obs  # ToolMessage
                ])
            # Find the best_guess tool call
            best_guess_tool_call = next(
                (tc for tc in tool_calls if tc.tool_calls[0]["name"] == "best_guess"),
                None
            )
            # If we didn't find a best guess tool call, we raise an error
            if not best_guess_tool_call:
                raise RuntimeError("No best_guess tool call found in the agent's scratchpad.")
            final_answer_call = best_guess_tool_call.tool_calls[0]
            final_answer = final_answer_call["args"]["answer"]
        
        # Add to chat history
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])
        # return the final answer and tools used
        return final_answer_call

# Initialize agent executor
agent_executor = CustomAgentExecutor(max_iterations=10)  