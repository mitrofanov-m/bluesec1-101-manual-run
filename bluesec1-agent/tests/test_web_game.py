import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bluesec1_client import PublicTask, TaskResult, ToolCallResult
from bluesec1_web.app import create_app
from bluesec1_web.game import GameStateError, InvestigationGame

TOOL_CATALOG = [
    {
        "name": "get_entity",
        "description": "Read one entity.",
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
        "name": "finish_investigation",
        "description": "Submit the verdict.",
        "parameters": {
            "type": "object",
            "properties": {"submission": {"type": "object"}},
            "required": ["submission"],
        },
    },
]


class FakeSession:
    def __init__(self) -> None:
        self.task = PublicTask(
            task_id="task-01",
            observation={
                "alert_text": json.dumps(
                    {
                        "alert_description": "Suspicious process started.",
                        "trigger_entities": [{"id": "entity-01", "type": "windows_process"}],
                    }
                ),
                "available_tools": json.dumps(TOOL_CATALOG),
            },
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.aborted = False

    @property
    def task_id(self) -> str:
        return self.task.task_id

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((tool_name, arguments))
        terminal = tool_name == "finish_investigation"
        task_result = (
            TaskResult(
                task_id=self.task_id,
                completion_reason="terminated",
                total_reward=4.2,
                tool_calls=len(self.calls),
            )
            if terminal
            else None
        )
        return ToolCallResult(
            task_id=self.task_id,
            tool_name=tool_name,
            observation={"last_tool_result": {"entity_id": "entity-01"}},
            reward=1.0,
            terminated=terminal,
            info={"tool_status": "ok", "tool_result": {"entity_id": "entity-01"}},
            task_result=task_result,
        )

    async def abort_task(self, *, error_message: str | None = None) -> TaskResult:
        self.aborted = True
        return TaskResult(task_id=self.task_id, completion_reason="aborted")


class FakeClient:
    def __init__(self, session: FakeSession, **_: Any) -> None:
        self.session = session
        self.run = SimpleNamespace(run_id="run-01")
        self.start_task_calls = 0
        self.closed = False

    async def open(self) -> SimpleNamespace:
        return self.run

    async def start_task(self) -> FakeSession | None:
        self.start_task_calls += 1
        return self.session if self.start_task_calls == 1 else None

    async def aclose(self, **_: Any) -> None:
        self.closed = True


def make_game() -> tuple[InvestigationGame, FakeClient, FakeSession]:
    session = FakeSession()
    client = FakeClient(session)
    game = InvestigationGame(client_factory=lambda **_: client)  # type: ignore[arg-type]
    return game, client, session


def start(game: InvestigationGame) -> dict[str, Any]:
    return asyncio.run(
        game.start(
            endpoint="example.test:443",
            token="secret-token",
            arena=None,
            agent_name="manual-test",
        )
    )


def test_terminal_tool_closes_run_without_leasing_another_task() -> None:
    game, client, session = make_game()
    initial = start(game)

    assert initial["phase"] == "active"
    assert initial["task_id"] == "task-01"
    assert [tool["name"] for tool in initial["available_tools"]] == [
        "get_entity",
        "finish_investigation",
    ]

    after_entity = asyncio.run(
        game.call_tool(
            "get_entity",
            {"entity_id": "entity-01", "reasoning": "Inspect the trigger."},
        )
    )
    assert after_entity["phase"] == "active"

    finished = asyncio.run(
        game.call_tool(
            "finish_investigation",
            {"submission": {"verdict": "benign", "legitimacy_evidence": []}},
        )
    )

    assert finished["phase"] == "completed"
    assert finished["task_result"]["completion_reason"] == "terminated"
    assert client.start_task_calls == 1
    assert client.closed is True
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]

    with pytest.raises(GameStateError, match="единственную задачу"):
        start(game)
    assert client.start_task_calls == 1


def test_abort_closes_the_only_task_without_a_verdict() -> None:
    game, client, session = make_game()
    start(game)

    state = asyncio.run(game.abort())

    assert state["phase"] == "aborted"
    assert state["task_result"]["completion_reason"] == "aborted"
    assert state["investigation_tool_calls"] == 0
    assert session.aborted is True
    assert client.closed is True
    assert client.start_task_calls == 1


def test_unadvertised_tool_is_rejected_locally() -> None:
    game, _, session = make_game()
    start(game)

    with pytest.raises(GameStateError, match="не объявлен"):
        asyncio.run(game.call_tool("delete_everything", {}))

    assert session.calls == []


def test_browser_snapshot_never_contains_the_runtime_token() -> None:
    game, _, _ = make_game()
    state = start(game)

    assert "secret-token" not in json.dumps(state)


def test_http_api_serves_page_and_browser_safe_task_state() -> None:
    game, client, _ = make_game()

    with TestClient(create_app(game)) as browser:
        page = browser.get("/")
        response = browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "secret-token",
                "arena": None,
                "agent_name": "manual-test",
            },
        )

        assert page.status_code == 200
        assert "BlueSec Manual Lab" in page.text
        assert response.status_code == 200
        assert response.json()["task_id"] == "task-01"
        assert "secret-token" not in response.text
        assert client.start_task_calls == 1
