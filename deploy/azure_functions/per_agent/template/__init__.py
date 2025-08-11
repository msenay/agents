import logging
import azure.functions as func
import json

from deploy.shared.agent_registry import build_agent


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # This function is stamped per agent; the name comes from app settings or path
        # For a template, allow ?agent=name override
        agent_name = req.params.get('agent') or 'default'
        try:
            query = req.params.get('q')
        except Exception:
            query = None
        if not query:
            body = req.get_json(silent=True) or {}
            query = body.get('q') or body.get('input')
        if not query:
            return func.HttpResponse("Missing 'q' or 'input'", status_code=400)

        agent = build_agent(agent_name)
        result = agent.invoke(query)
        content = result["messages"][-1].content if result and "messages" in result else ""
        return func.HttpResponse(content, status_code=200)
    except Exception as e:
        logging.exception("per-agent function error")
        return func.HttpResponse(str(e), status_code=500)



