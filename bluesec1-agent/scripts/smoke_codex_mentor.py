"""Run one credential-free live smoke check through the local Codex CLI."""

from __future__ import annotations

import argparse
import asyncio
import json

from bluesec1_web.mentor import MentorConfig, MentorService


async def main(model: str) -> None:
    mentor = MentorService(
        MentorConfig(provider="codex", model=model, reasoning_effort="medium")
    )
    context = {
        "phase": "active",
        "initial_observation": {
            "alert_text": "PowerShell started with an encoded command on host WS-17."
        },
        "current_observation": {
            "last_tool_result": {
                "entity_id": "process-17",
                "image_path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "parent_process": "winword.exe",
            },
        },
        "history": [],
        "available_tools": [{"name": "get_entity"}, {"name": "get_relation"}],
    }
    tools = [
        {
            "name": "get_entity",
            "description": "Read one entity by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["entity_id", "reasoning"],
            },
        },
        {
            "name": "get_relation",
            "description": "Read one relation by ID.",
            "parameters": {
                "type": "object",
                "properties": {"relation_id": {"type": "string"}},
                "required": ["relation_id"],
            },
        },
    ]
    event = {
        "kind": "tool_call",
        "tool_name": "get_entity",
        "arguments": {"entity_id": "process-17"},
        "response": {"observation": {"last_tool_result": context["current_observation"]}},
    }
    results = {
        "advice": (await mentor.advice(context)).model_dump(mode="json"),
        "next_step": (await mentor.next_step(context, tools)).model_dump(mode="json"),
        "explanation": (await mentor.explain(context, event)).model_dump(mode="json"),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-luna")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.model))
