# This code is configuring Cross-Origin Resource Sharing (CORS) for a FastAPI (or Starlette) application. Here’s what’s happening step by step:

# High-Level Overview
# app.add_middleware(...) adds a middleware component to your web application.
# CORSMiddleware is a middleware that controls how your API handles requests from different origins (domains), which is important for security when your frontend and backend are on different servers.

# CORSMiddleware: This middleware handles CORS, which is a browser security feature that restricts web pages from making requests to a different domain than the one that served the web page.
# allow_origins=["http://localhost:3000"]: Only allows requests from this origin (your frontend, likely running on React or similar).
# allow_credentials=True: Allows cookies, authorization headers, or TLS client certificates to be included in requests.
# allow_methods=["*"]: Allows all HTTP methods (GET, POST, PUT, DELETE, etc.).
# allow_headers=["*"]: Allows all headers in requests.

# How It Works Internally
# The add_middleware function:
# Checks if the middleware stack has already been built (i.e., if the app has started). If so, it raises an error to prevent changes after startup.
# Wraps the CORSMiddleware with the provided arguments and inserts it into the middleware stack.

# Why Use This?
# Security: Prevents unauthorized domains from accessing your API.
# Development Convenience: Allows your frontend (often running on a different port during development) to communicate with your backend.

# Key points
# If you deploy your frontend to a different URL, you’ll need to update allow_origins.
# Using "*" for allow_origins is not recommended in production if you use allow_credentials=True.

import asyncio

from agent import QueueCallbackHandler, agent_executor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# initilizing our application
app = FastAPI()

# CORS Middleware Flow:
# [Frontend (http://localhost:3000)] --(HTTP Request)--> [FastAPI App]
#        |
#        |---> [CORSMiddleware]
#                 |
#                 |---> Checks Origin, Methods, Headers
#                 |---> If allowed: Forwards request to route handler
#                 |---> If not allowed: Returns CORS error
#        |
#        |<--- [Response with CORS headers if allowed]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True  # set to True to see verbose output in console
    ))
    # initialize various components to stream
    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                # send end of step token
                yield "</step>"
            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    # send start of step token followed by step name tokens
                    yield f"<step><step_name>{tool_name}</step_name>"
                if tool_args := tool_calls[0]["function"]["arguments"]:
                    # tool args are streamed directly, ensure it's properly encoded
                    yield tool_args
        except Exception as e:
            print(f"Error streaming token: {e}")
            continue
    await task

# invoke function
@app.post("/invoke")
async def invoke(content: str):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)
    # return the streaming response
    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
