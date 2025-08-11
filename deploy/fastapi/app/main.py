#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from deploy.shared.agent_registry import build_agent

app = FastAPI(title="Agents API")


class InvokeBody(BaseModel):
    agent: str
    q: Optional[str] = None
    input: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
def agents():
    return {"agents": ["support", "sales", "default"]}


@app.post("/invoke")
def invoke(body: InvokeBody):
    text = body.q or body.input
    if not text:
        raise HTTPException(status_code=400, detail="Missing q or input")
    agent = build_agent(body.agent)
    result = agent.invoke(text)
    content = result["messages"][-1].content if result and "messages" in result else ""
    return {"agent": body.agent, "output": content}



