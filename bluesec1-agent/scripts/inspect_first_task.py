"""Lease the first BlueSec task, inspect it, and abort without a verdict."""

import asyncio
import json
import os

import truststore

from bluesec1_client import RemoteBenchmarkClient


def _tool_names(observation: dict[str, object]) -> list[str]:
    raw_tools = observation.get("available_tools")
    if not isinstance(raw_tools, str):
        return []
    tools = json.loads(raw_tools)
    return [tool["name"] for tool in tools if isinstance(tool, dict) and "name" in tool]


def _visible_observation(observation: dict[str, object]) -> dict[str, object]:
    """Return the initial observation without the verbose tool schemas."""
    return {key: value for key, value in observation.items() if key != "available_tools"}


async def main() -> None:
    truststore.inject_into_ssl()
    token = os.environ["SCENARIO_RUNTIME_TOKEN"]
    endpoint = os.getenv("SCENARIO_RUNTIME_ENDPOINT", "bluesec.team:443")
    arena = os.getenv("SCENARIO_RUNTIME_ARENA")

    async with RemoteBenchmarkClient(
        endpoint=endpoint,
        token=token,
        verify_tls=True,
        run_label="inspect-first-task",
        agent_name="bluesec-101-human-inspection",
        arena=arena,
        metadata={"purpose": "inspect-first-task-without-verdict"},
    ) as client:
        session = await client.start_task()
        if session is None:
            print(json.dumps({"status": "queue_exhausted"}, indent=2))
            return

        print(
            json.dumps(
                {
                    "status": "first_task_leased",
                    "task_id": session.task_id,
                    "observation": _visible_observation(session.task.observation),
                    "available_tools": _tool_names(session.task.observation),
                    "investigation_tool_calls": 0,
                    "verdict_submitted": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        result = await session.abort_task(
            error_message=(
                "Stopped intentionally after inspecting the first task; no verdict submitted."
            )
        )
        print(
            json.dumps(
                {
                    "status": "first_task_aborted",
                    "completion_reason": result.completion_reason,
                    "steps_taken": result.steps_taken,
                    "tool_calls": result.tool_calls,
                    "total_reward": result.total_reward,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
