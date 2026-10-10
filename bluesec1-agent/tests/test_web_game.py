import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bluesec1_client import PublicTask, RemoteRunNotFoundError, TaskResult, ToolCallResult
from bluesec1_web.app import create_app
from bluesec1_web.game import GameStateError, InvestigationGame
from bluesec1_web.mentor import (
    AdviceReply,
    CloseReply,
    ExplanationReply,
    HelpReply,
    MentorConfig,
    evidence_call_fingerprint,
)

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
    def __init__(self, task_id: str = "task-01") -> None:
        self.task = PublicTask(
            task_id=task_id,
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


class MissingRunSession(FakeSession):
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raise RemoteRunNotFoundError("The remote run could not be found.")

    async def abort_task(self, *, error_message: str | None = None) -> TaskResult:
        raise RemoteRunNotFoundError("The remote run could not be found.")


class FakeClient:
    def __init__(self, session: FakeSession, *, run_id: str = "run-01", **_: Any) -> None:
        self.session = session
        self.run = SimpleNamespace(run_id=run_id)
        self.start_task_calls = 0
        self.closed = False

    async def open(self) -> SimpleNamespace:
        return self.run

    async def start_task(self) -> FakeSession | None:
        self.start_task_calls += 1
        return self.session if self.start_task_calls == 1 else None

    async def aclose(self, **_: Any) -> None:
        self.closed = True


class FakeClientFactory:
    def __init__(self) -> None:
        self.clients: list[FakeClient] = []
        self.sessions: list[FakeSession] = []

    def __call__(self, **_: Any) -> FakeClient:
        sequence = len(self.clients) + 1
        session = FakeSession(task_id=f"task-{sequence:02d}")
        client = FakeClient(session, run_id=f"run-{sequence:02d}")
        self.sessions.append(session)
        self.clients.append(client)
        return client


class FakeMentor:
    def __init__(self, config: MentorConfig) -> None:
        self.config = config
        self.received_tools: list[dict[str, Any]] = []
        self.received_contexts: list[dict[str, Any]] = []

    @property
    def public_config(self) -> dict[str, str]:
        return self.config.public()

    async def advice(self, context: dict[str, Any]) -> AdviceReply:
        assert context["available_tool_names"] == ["get_entity", "finish_investigation"]
        self.received_contexts.append(context)
        return AdviceReply(
            title="Не паникуем",
            summary="Алерт пока только гипотеза.",
            details=["Проверьте исходный процесс."],
        )

    async def next_step(
        self,
        context: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> HelpReply:
        self.received_contexts.append(context)
        self.received_tools = tools
        return HelpReply(
            action="call_tool",
            explanation="Сначала читаем исходную сущность.",
            tool_name="get_entity",
            arguments_json=json.dumps(
                {"entity_id": "entity-01", "reasoning": "Inspect the trigger."}
            ),
        )

    async def prepare_close(
        self,
        context: dict[str, Any],
        tool: dict[str, Any],
    ) -> CloseReply:
        self.received_contexts.append(context)
        assert tool["name"] == "finish_investigation"
        return CloseReply(
            action="prepare_close",
            explanation="Доказательств достаточно для проверки вердикта.",
            arguments_json=json.dumps({"submission": {}}),
        )

    async def explain(
        self,
        context: dict[str, Any],
        event: dict[str, Any],
    ) -> ExplanationReply:
        assert event["tool_name"] == "get_entity"
        return ExplanationReply(
            meaning="Инструмент нашёл сущность.",
            important=["entity_id связывает следующие шаги."],
            next_hint="Проверьте новые связи.",
        )


class FakeMentorFactory:
    def __init__(self) -> None:
        self.instances: list[FakeMentor] = []

    def __call__(self, config: MentorConfig) -> FakeMentor:
        mentor = FakeMentor(config)
        self.instances.append(mentor)
        return mentor


def make_game() -> tuple[InvestigationGame, FakeClient, FakeSession]:
    session = FakeSession()
    client = FakeClient(session)
    game = InvestigationGame(client_factory=lambda **_: client)  # type: ignore[arg-type]
    return game, client, session


def make_missing_run_game() -> tuple[InvestigationGame, FakeClient, MissingRunSession]:
    session = MissingRunSession()
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


def test_restart_aborts_active_task_and_immediately_opens_new_run() -> None:
    factory = FakeClientFactory()
    game = InvestigationGame(client_factory=factory)  # type: ignore[arg-type]
    initial = start(game)

    restarted = asyncio.run(game.restart())

    assert initial["run_id"] == "run-01"
    assert initial["task_id"] == "task-01"
    assert factory.sessions[0].aborted is True
    assert factory.clients[0].closed is True
    assert restarted["phase"] == "active"
    assert restarted["run_id"] == "run-02"
    assert restarted["task_id"] == "task-02"
    assert restarted["history"][0]["kind"] == "task_started"
    assert len(factory.clients) == 2


def test_finish_study_aborts_active_task_and_returns_to_ready_state() -> None:
    game, client, session = make_game()
    start(game)

    ready = asyncio.run(game.finish_study())

    assert session.aborted is True
    assert client.closed is True
    assert ready["phase"] == "ready"
    assert ready["can_start"] is True
    assert ready["ever_started"] is False
    assert ready["run_id"] is None
    assert ready["task_id"] is None
    assert ready["history"] == []


def test_missing_remote_run_becomes_restartable_terminal_state() -> None:
    game, client, _ = make_missing_run_game()
    start(game)

    stopped = asyncio.run(game.abort())

    assert stopped["phase"] == "run_stopped"
    assert stopped["can_restart"] is True
    assert stopped["can_start"] is False
    assert "остановлен средой" in stopped["last_error"]
    assert stopped["history"][-1]["kind"] == "run_stopped"
    assert client.closed is True

    ready = asyncio.run(game.reset_stopped_run())

    assert ready["phase"] == "ready"
    assert ready["can_start"] is True
    assert ready["can_restart"] is False
    assert ready["run_id"] is None
    assert ready["task_id"] is None
    assert ready["history"] == []


def test_missing_remote_run_during_tool_call_does_not_return_500() -> None:
    game, _, _ = make_missing_run_game()

    with TestClient(create_app(game)) as browser:
        started = browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "secret-token",
                "arena": None,
                "agent_name": "manual-test",
            },
        )
        response = browser.post(
            "/api/tool",
            json={
                "tool_name": "get_entity",
                "arguments": {"entity_id": "entity-01", "reasoning": "Inspect it."},
            },
        )

    assert started.status_code == 200
    assert response.status_code == 200
    assert response.json()["phase"] == "run_stopped"
    assert response.json()["can_restart"] is True


def test_missing_remote_run_during_abort_does_not_return_500() -> None:
    game, _, _ = make_missing_run_game()

    with TestClient(create_app(game)) as browser:
        browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "secret-token",
                "arena": None,
                "agent_name": "manual-test",
            },
        )
        response = browser.post("/api/session/abort", json={})

    assert response.status_code == 200
    assert response.json()["phase"] == "run_stopped"
    assert response.json()["can_restart"] is True


def test_button_driven_mentor_endpoints_use_secret_free_state() -> None:
    session = FakeSession()
    client = FakeClient(session)
    mentor_factory = FakeMentorFactory()
    game = InvestigationGame(
        client_factory=lambda **_: client,  # type: ignore[arg-type]
        mentor_factory=mentor_factory,
    )

    with TestClient(create_app(game)) as browser:
        started = browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "runtime-secret",
                "arena": None,
                "agent_name": "manual-test",
                "mentor": {
                    "provider": "openai",
                    "endpoint": "https://api.example.test/v1",
                    "api_key": "llm-secret",
                    "model": "small-test-model",
                    "reasoning_effort": "medium",
                },
            },
        )
        advice = browser.post("/api/mentor/advice", json={})
        next_step = browser.post("/api/mentor/help", json={})
        tool_result = browser.post(
            "/api/tool",
            json={
                "tool_name": next_step.json()["tool_name"],
                "arguments": next_step.json()["arguments"],
            },
        )
        explanation = browser.post("/api/mentor/explain", json={"history_index": 1})
        close_draft = browser.post("/api/mentor/close", json={})

    assert started.status_code == 200
    assert started.json()["mentor"] == {
        "provider": "openai",
        "model": "small-test-model",
        "reasoning_effort": "medium",
    }
    assert "llm-secret" not in json.dumps(started.json())
    assert "runtime-secret" not in json.dumps(started.json())
    assert advice.json()["title"] == "Не паникуем"
    assert next_step.json()["tool_name"] == "get_entity"
    assert next_step.json()["action"] == "call_tool"
    assert tool_result.status_code == 200
    assert explanation.json()["meaning"] == "Инструмент нашёл сущность."
    assert close_draft.json() == {
        "action": "prepare_close",
        "explanation": "Доказательств достаточно для проверки вердикта.",
        "tool_name": "finish_investigation",
        "arguments": {"submission": {}},
    }
    assert [tool["name"] for tool in mentor_factory.instances[0].received_tools] == ["get_entity"]
    assert mentor_factory.instances[0].received_contexts[1]["conversation_history"] == [
        {
            "kind": "advice",
            "title": "Не паникуем",
            "summary": "Алерт пока только гипотеза.",
            "details": ["Проверьте исходный процесс."],
        }
    ]
    assert mentor_factory.instances[0].received_contexts[2]["conversation_history"][1] == {
        "kind": "help",
        "action": "call_tool",
        "explanation": "Сначала читаем исходную сущность.",
    }
    assert mentor_factory.instances[0].received_contexts[2]["completed_calls"][0] == {
        "tool_name": "get_entity",
        "fingerprint": evidence_call_fingerprint(
            "get_entity",
            {"entity_id": "entity-01", "reasoning": "Inspect the trigger."},
        ),
        "status": "ok",
    }
    assert mentor_factory.instances[0].received_contexts[2]["latest_tool_call"] == {
        "tool_name": "get_entity",
        "arguments": {"entity_id": "entity-01", "reasoning": "Inspect the trigger."},
        "completed": True,
        "status": "ok",
        "result": {"entity_id": "entity-01"},
        "reward": 1.0,
        "terminated": False,
        "error": None,
    }
    assert mentor_factory.instances[0].received_contexts[2]["attention_tool_fields"] == [
        {"field": "entity_id", "value": "entity-01", "source": "get_entity#1"},
        {"field": "id", "value": "entity-01", "source": "initial_observation"},
        {"field": "type", "value": "windows_process", "source": "initial_observation"},
    ]


def test_mentor_context_keeps_only_latest_full_tool_result() -> None:
    game, _, _ = make_game()
    game._initial_observation = {  # noqa: SLF001 - focused context serialization test
        "alert_text": json.dumps({"entity_id": "alert-entity", "type": "windows_process"}),
        "available_tools": json.dumps(TOOL_CATALOG),
    }
    game._tool_specs = TOOL_CATALOG  # noqa: SLF001
    game._history = [  # noqa: SLF001
        {
            "kind": "tool_call",
            "tool_name": "get_entity",
            "arguments": {"entity_id": "old-entity", "reasoning": "Read old."},
            "response": {
                "observation": {
                    "last_tool_result": {
                        "entity_id": "old-entity",
                        "hostname": "old-host",
                        "blob": "old-heavy-payload" * 1_000,
                    }
                },
                "info": {"tool_status": "ok"},
                "terminated": False,
            },
        },
        {
            "kind": "tool_call",
            "tool_name": "get_entity",
            "arguments": {"entity_id": "new-entity", "reasoning": "Read new."},
            "response": {
                "observation": {
                    "last_tool_result": {
                        "entity_id": "new-entity",
                        "cmdline": "powershell.exe -enc ...",
                    }
                },
                "info": {"tool_status": "ok"},
                "terminated": False,
            },
        },
    ]
    game._remember_mentor_message(  # noqa: SLF001
        "help",
        {
            "action": "call_tool",
            "explanation": "Проверим новую сущность.",
            "tool_name": "get_entity",
            "arguments": {"entity_id": "new-entity", "reasoning": "Read new."},
        },
    )

    context = game._mentor_context()  # noqa: SLF001

    assert context["conversation_history"] == [
        {
            "kind": "help",
            "action": "call_tool",
            "explanation": "Проверим новую сущность.",
        }
    ]
    assert context["completed_calls"] == [
        {
            "tool_name": "get_entity",
            "fingerprint": evidence_call_fingerprint(
                "get_entity", {"entity_id": "old-entity", "reasoning": "Read old."}
            ),
            "status": "ok",
        },
        {
            "tool_name": "get_entity",
            "fingerprint": evidence_call_fingerprint(
                "get_entity", {"entity_id": "new-entity", "reasoning": "Read new."}
            ),
            "status": "ok",
        },
    ]
    assert context["latest_tool_call"]["result"] == {
        "entity_id": "new-entity",
        "cmdline": "powershell.exe -enc ...",
    }
    assert {fact["value"] for fact in context["attention_tool_fields"]} >= {
        "old-entity",
        "old-host",
        "new-entity",
        "alert-entity",
    }
    rendered_context = json.dumps(context)
    assert "old-heavy-payload" not in rendered_context
    assert '"arguments"' in json.dumps(context["latest_tool_call"])
    assert all("arguments" not in call for call in context["completed_calls"])
    assert len(rendered_context) < 5_000


def test_openai_mentor_requires_endpoint_and_key() -> None:
    game, _, _ = make_game()

    with TestClient(create_app(game)) as browser:
        response = browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "secret-token",
                "mentor": {"provider": "openai", "model": "small-test-model"},
            },
        )

    assert response.status_code == 422


def test_restart_http_endpoint_returns_new_run_and_task() -> None:
    factory = FakeClientFactory()
    game = InvestigationGame(client_factory=factory)  # type: ignore[arg-type]

    with TestClient(create_app(game)) as browser:
        browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "secret-token",
                "arena": None,
                "agent_name": "manual-test",
            },
        )
        response = browser.post("/api/session/restart", json={})

    assert response.status_code == 200
    assert response.json()["phase"] == "active"
    assert response.json()["run_id"] == "run-02"
    assert response.json()["task_id"] == "task-02"
    assert factory.sessions[0].aborted is True


def test_finish_study_http_endpoint_returns_to_start_screen() -> None:
    game, _, session = make_game()

    with TestClient(create_app(game)) as browser:
        browser.post(
            "/api/session/start",
            json={
                "endpoint": "example.test:443",
                "token": "secret-token",
                "arena": None,
                "agent_name": "manual-test",
            },
        )
        response = browser.post("/api/session/end", json={})

    assert response.status_code == 200
    assert response.json()["phase"] == "ready"
    assert response.json()["can_start"] is True
    assert session.aborted is True


def test_unadvertised_tool_is_rejected_locally() -> None:
    game, _, session = make_game()
    start(game)

    with pytest.raises(GameStateError, match="не объявлен"):
        asyncio.run(game.call_tool("delete_everything", {}))

    assert session.calls == []


def test_browser_snapshot_never_contains_the_runtime_token() -> None:
    factory = FakeClientFactory()
    game = InvestigationGame(client_factory=factory)  # type: ignore[arg-type]
    start(game)
    state = asyncio.run(game.restart())

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
        assert "Документация" in page.text
        assert "О приложении" in page.text
        assert "Инструменты расследования" in page.text
        assert "SIEMён Петрович" in page.text
        assert "Помоги закрыть" in page.text
        assert "История, алерт и другие вызовы в этот запрос не передаются" in page.text
        assert response.status_code == 200
        assert response.json()["task_id"] == "task-01"
        assert "secret-token" not in response.text
        assert client.start_task_calls == 1
