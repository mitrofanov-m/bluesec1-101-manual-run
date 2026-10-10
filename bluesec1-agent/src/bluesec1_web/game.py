"""Stateful one-task bridge between the browser and the BlueSec runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bluesec1_client import (
    RemoteBenchmarkClient,
    RemoteBenchmarkSession,
    RemoteRunNotFoundError,
)


class GameStateError(RuntimeError):
    """A safe, user-facing lifecycle or input error."""

    def __init__(self, message: str, *, code: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


ClientFactory = Callable[..., RemoteBenchmarkClient]


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    """Runtime connection details retained only in local server memory."""

    endpoint: str
    token: str
    arena: str | None
    agent_name: str


class InvestigationGame:
    """Own exactly one remote run and lease at most one task from it."""

    def __init__(self, client_factory: ClientFactory = RemoteBenchmarkClient) -> None:
        self._client_factory = client_factory
        self._lock = asyncio.Lock()
        self._client: RemoteBenchmarkClient | None = None
        self._session: RemoteBenchmarkSession | None = None
        self._phase = "ready"
        self._ever_leased_task = False
        self._run_id: str | None = None
        self._task_id: str | None = None
        self._initial_observation: dict[str, Any] | None = None
        self._current_observation: dict[str, Any] | None = None
        self._tool_specs: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._task_result: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._close_warning: str | None = None
        self._connection_settings: ConnectionSettings | None = None

    async def start(
        self,
        *,
        endpoint: str,
        token: str,
        arena: str | None,
        agent_name: str,
    ) -> dict[str, Any]:
        """Create one run and lease its first task, never a second task."""
        async with self._lock:
            if self._ever_leased_task:
                raise GameStateError(
                    "Эта игровая сессия уже получила свою единственную задачу. "
                    "Перезапустите локальный сервер, чтобы начать новый run.",
                    code="one_task_limit_reached",
                )
            if self._phase == "connecting":
                raise GameStateError(
                    "Подключение уже выполняется.",
                    code="connection_in_progress",
                )
            if self._client is not None:
                raise GameStateError(
                    "Run уже создан.",
                    code="run_already_started",
                )
            return await self._start_run_locked(
                ConnectionSettings(
                    endpoint=endpoint,
                    token=token,
                    arena=arena,
                    agent_name=agent_name,
                )
            )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call one server-advertised tool and retain the complete response."""
        async with self._lock:
            session = self._require_active_session()
            advertised_names = {spec["name"] for spec in self._tool_specs}
            if tool_name not in advertised_names:
                raise GameStateError(
                    f"Инструмент {tool_name!r} не объявлен средой для этой задачи.",
                    code="tool_not_advertised",
                    status_code=400,
                )
            if not isinstance(arguments, dict):
                raise GameStateError(
                    "Аргументы инструмента должны быть JSON-объектом.",
                    code="invalid_tool_arguments",
                    status_code=400,
                )

            event: dict[str, Any] = {
                "kind": "tool_call",
                "at": self._now(),
                "tool_name": tool_name,
                "arguments": arguments,
            }
            try:
                response = await session.call_tool(tool_name, arguments)
            except RemoteRunNotFoundError:
                event["error"] = self._run_stopped_message()
                self._history.append(event)
                await self._mark_run_stopped_locked()
                return self.snapshot()
            except Exception as error:
                event["error"] = f"{type(error).__name__}: {error}"
                self._history.append(event)
                raise

            response_payload = response.model_dump(mode="json")
            event["response"] = response_payload
            self._history.append(event)

            if response.observation:
                self._current_observation = dict(response.observation)
                if "available_tools" in response.observation:
                    self._tool_specs = self._decode_tool_specs(response.observation)

            if response.task_result is not None:
                self._task_result = response.task_result.model_dump(mode="json")
                self._phase = "completed"
                await self._close_run_locked()

            return self.snapshot()

    async def abort(self) -> dict[str, Any]:
        """Abort the only task without submitting a verdict."""
        async with self._lock:
            session = self._require_active_session()
            try:
                result = await session.abort_task(
                    error_message="Stopped by the user in the local manual investigation interface."
                )
            except RemoteRunNotFoundError:
                self._history.append(
                    {
                        "kind": "run_stopped",
                        "at": self._now(),
                        "error": self._run_stopped_message(),
                    }
                )
                await self._mark_run_stopped_locked()
                return self.snapshot()
            self._task_result = result.model_dump(mode="json")
            self._history.append(
                {
                    "kind": "task_aborted",
                    "at": self._now(),
                    "task_result": self._task_result,
                }
            )
            self._phase = "aborted"
            await self._close_run_locked()
            return self.snapshot()

    async def reset_stopped_run(self) -> dict[str, Any]:
        """Forget a remotely stopped run so the user may explicitly create another."""
        async with self._lock:
            if self._phase != "run_stopped":
                raise GameStateError(
                    "Новый run можно начать здесь только после остановки текущего run средой.",
                    code="run_not_stopped",
                )
            await self._close_run_locked()
            self._reset_state_locked(clear_connection=True)
            return self.snapshot()

    async def restart(self) -> dict[str, Any]:
        """Stop the current task when needed and immediately open a new run."""
        async with self._lock:
            if self._phase == "connecting":
                raise GameStateError(
                    "Дождитесь завершения текущего подключения.",
                    code="connection_in_progress",
                )
            settings = self._connection_settings
            if settings is None:
                raise GameStateError(
                    "Параметры подключения не сохранены. Начните с формы подключения.",
                    code="connection_settings_missing",
                )
            if self._phase == "active" and self._session is not None:
                with suppress(RemoteRunNotFoundError):
                    await self._session.abort_task(
                        error_message="Restarted by the user in the local manual interface."
                    )
            await self._close_run_locked()
            self._reset_state_locked(clear_connection=False)
            return await self._start_run_locked(settings)

    async def finish_study(self) -> dict[str, Any]:
        """Close any live task and forget the connection before showing the start screen."""
        async with self._lock:
            if self._phase == "connecting":
                raise GameStateError(
                    "Дождитесь завершения текущего подключения.",
                    code="connection_in_progress",
                )
            if self._phase == "active" and self._session is not None:
                with suppress(RemoteRunNotFoundError):
                    await self._session.abort_task(
                        error_message="Study finished by the user in the local manual interface."
                    )
            await self._close_run_locked()
            self._reset_state_locked(clear_connection=True)
            return self.snapshot()

    async def shutdown(self) -> None:
        """Release a live task if the local web server stops mid-investigation."""
        async with self._lock:
            if self._phase == "active" and self._session is not None:
                try:
                    result = await self._session.abort_task(
                        error_message="Local manual investigation server stopped before a verdict."
                    )
                    self._task_result = result.model_dump(mode="json")
                    self._phase = "aborted"
                except Exception as error:  # pragma: no cover - shutdown is best effort
                    self._close_warning = f"{type(error).__name__}: {error}"
            await self._close_run_locked()

    def snapshot(self) -> dict[str, Any]:
        """Return browser-safe state without credentials or transport objects."""
        return {
            "phase": self._phase,
            "one_task_only": True,
            "ever_started": self._ever_leased_task,
            "can_start": not self._ever_leased_task and self._client is None,
            "can_restart": self._phase == "run_stopped",
            "run_id": self._run_id,
            "task_id": self._task_id,
            "initial_observation": self._initial_observation,
            "current_observation": self._current_observation,
            "available_tools": self._tool_specs,
            "history": self._history,
            "task_result": self._task_result,
            "last_error": self._last_error,
            "close_warning": self._close_warning,
            "investigation_tool_calls": sum(
                item["kind"] == "tool_call" for item in self._history
            ),
        }

    def _require_active_session(self) -> RemoteBenchmarkSession:
        if self._phase != "active" or self._session is None:
            raise GameStateError(
                "Активной задачи нет или расследование уже закончено.",
                code="task_not_active",
            )
        return self._session

    async def _start_run_locked(self, settings: ConnectionSettings) -> dict[str, Any]:
        """Open one run from retained settings while the game lock is held."""
        self._phase = "connecting"
        self._last_error = None
        client = self._client_factory(
            endpoint=settings.endpoint,
            token=settings.token,
            verify_tls=True,
            run_label="bluesec-manual-lab",
            agent_name=settings.agent_name,
            model_name="human-in-the-loop",
            arena=settings.arena,
            metadata={
                "interface": "bluesec-manual-lab",
                "mode": "single-task-human-investigation",
            },
        )
        self._client = client

        try:
            remote_run = await client.open()
            self._connection_settings = settings
            self._run_id = remote_run.run_id
            session = await client.start_task()
        except Exception:
            self._phase = "ready"
            self._client = None
            await self._best_effort_close(client)
            raise

        if session is None:
            self._ever_leased_task = True
            self._phase = "empty"
            self._client = None
            await self._best_effort_close(client)
            return self.snapshot()

        self._session = session
        self._ever_leased_task = True
        self._phase = "active"
        self._task_id = session.task_id
        self._initial_observation = dict(session.task.observation)
        self._current_observation = dict(session.task.observation)
        try:
            self._tool_specs = self._decode_tool_specs(session.task.observation)
        except Exception:
            try:
                await session.abort_task(
                    error_message="Runtime returned an invalid tool catalog to the web client."
                )
            finally:
                await self._close_run_locked()
            raise
        self._history.append(
            {
                "kind": "task_started",
                "at": self._now(),
                "task_id": session.task_id,
                "observation": dict(session.task.observation),
            }
        )
        return self.snapshot()

    async def _close_run_locked(self) -> None:
        client = self._client
        self._client = None
        self._session = None
        if client is not None:
            try:
                await client.aclose(reason="client_requested")
            except Exception as error:
                self._close_warning = f"{type(error).__name__}: {error}"

    async def _mark_run_stopped_locked(self) -> None:
        self._phase = "run_stopped"
        self._last_error = self._run_stopped_message()
        await self._close_run_locked()

    def _reset_state_locked(self, *, clear_connection: bool) -> None:
        self._phase = "ready"
        self._ever_leased_task = False
        self._run_id = None
        self._task_id = None
        self._initial_observation = None
        self._current_observation = None
        self._tool_specs = []
        self._history = []
        self._task_result = None
        self._last_error = None
        self._close_warning = None
        if clear_connection:
            self._connection_settings = None

    @staticmethod
    def _run_stopped_message() -> str:
        return (
            "Run был остановлен средой соревнования. Продолжить это расследование невозможно."
        )

    async def _best_effort_close(self, client: RemoteBenchmarkClient) -> None:
        try:
            await client.aclose(reason="client_error")
        except Exception as error:
            self._close_warning = f"{type(error).__name__}: {error}"

    @staticmethod
    def _decode_tool_specs(observation: dict[str, Any]) -> list[dict[str, Any]]:
        raw_specs = observation.get("available_tools")
        if isinstance(raw_specs, str):
            try:
                raw_specs = json.loads(raw_specs)
            except json.JSONDecodeError as error:
                raise GameStateError(
                    "Среда вернула некорректный available_tools.",
                    code="invalid_tool_catalog",
                    status_code=502,
                ) from error
        if not isinstance(raw_specs, list):
            raise GameStateError(
                "Среда не вернула каталог available_tools.",
                code="missing_tool_catalog",
                status_code=502,
            )

        specs: list[dict[str, Any]] = []
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, dict) or not isinstance(raw_spec.get("name"), str):
                raise GameStateError(
                    "Среда вернула неподдерживаемый формат инструмента.",
                    code="invalid_tool_catalog",
                    status_code=502,
                )
            spec = dict(raw_spec)
            if not isinstance(spec.get("parameters"), dict):
                spec["parameters"] = {"type": "object", "properties": {}}
            specs.append(spec)
        return specs

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
