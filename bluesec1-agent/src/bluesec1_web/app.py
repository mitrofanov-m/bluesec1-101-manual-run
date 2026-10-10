"""FastAPI entry point for the local BlueSec manual investigation UI."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import truststore
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from bluesec1_client import (
    CompetitionClosedError,
    RemoteAuthenticationError,
    RemoteAuthorizationError,
    RemoteRateLimitError,
    RemoteRequestError,
    RemoteRunNotFoundError,
    RemoteUnavailableError,
    RunCapacityExceededError,
)

from .game import GameStateError, InvestigationGame
from .mentor import MentorConfig, MentorError

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")


class MentorRequest(BaseModel):
    """LLM connection chosen on the local start screen."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["codex", "openai"] = "codex"
    endpoint: str | None = Field(default=None, max_length=512)
    api_key: SecretStr | None = None
    model: str = Field(default="gpt-5.6-luna", min_length=1, max_length=255)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"

    @model_validator(mode="after")
    def validate_provider_settings(self) -> MentorRequest:
        if self.provider == "openai":
            if not self.endpoint:
                raise ValueError("Для OpenAI API укажите endpoint.")
            if self.api_key is None or not self.api_key.get_secret_value():
                raise ValueError("Для OpenAI API укажите ключ.")
        return self

    def to_config(self) -> MentorConfig:
        return MentorConfig(
            provider=self.provider,
            endpoint=self.endpoint,
            api_key=self.api_key.get_secret_value() if self.api_key is not None else None,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )


class StartRequest(BaseModel):
    """Connection parameters supplied once by the local browser."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(default="bluesec.team:443", min_length=1, max_length=255)
    token: SecretStr = Field(min_length=1)
    arena: str | None = Field(default=None, min_length=1, max_length=64)
    agent_name: str = Field(default="bluesec-manual-lab", min_length=1, max_length=255)
    mentor: MentorRequest = Field(default_factory=MentorRequest)


class ToolRequest(BaseModel):
    """One dynamic call selected from the server-advertised catalog."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=255)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ExplainRequest(BaseModel):
    """Select one completed tool call from the retained trace."""

    model_config = ConfigDict(extra="forbid")

    history_index: int = Field(ge=0)


def create_app(game: InvestigationGame | None = None) -> FastAPI:
    """Build the local app with an injectable game controller for tests."""
    controller = game or InvestigationGame()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        truststore.inject_into_ssl()
        yield
        await controller.shutdown()

    app = FastAPI(
        title="BlueSec Manual Lab",
        description="One-task human investigation interface for BlueSec.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.game = controller
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.exception_handler(GameStateError)
    async def game_error_handler(_: Request, error: GameStateError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @app.exception_handler(MentorError)
    async def mentor_error_handler(_: Request, error: MentorError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": {"code": "mentor_error", "message": str(error)}},
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return controller.snapshot()

    @app.post("/api/session/start")
    async def start_session(payload: StartRequest) -> dict[str, Any]:
        try:
            return await controller.start(
                endpoint=payload.endpoint,
                token=payload.token.get_secret_value(),
                arena=payload.arena,
                agent_name=payload.agent_name,
                mentor=payload.mentor.to_config(),
            )
        except GameStateError:
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/tool")
    async def call_tool(payload: ToolRequest) -> dict[str, Any]:
        try:
            return await controller.call_tool(payload.tool_name, payload.arguments)
        except GameStateError:
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/mentor/advice")
    async def mentor_advice() -> dict[str, Any]:
        try:
            return await controller.mentor_advice()
        except (GameStateError, MentorError):
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/mentor/help")
    async def mentor_help() -> dict[str, Any]:
        try:
            return await controller.mentor_next_step()
        except (GameStateError, MentorError):
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/mentor/close")
    async def mentor_close() -> dict[str, Any]:
        try:
            return await controller.mentor_prepare_close()
        except (GameStateError, MentorError):
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/mentor/explain")
    async def mentor_explain(payload: ExplainRequest) -> dict[str, Any]:
        try:
            return await controller.mentor_explain(payload.history_index)
        except (GameStateError, MentorError):
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/session/abort")
    async def abort_session() -> dict[str, Any]:
        try:
            return await controller.abort()
        except GameStateError:
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/session/reset")
    async def reset_stopped_session() -> dict[str, Any]:
        return await controller.reset_stopped_run()

    @app.post("/api/session/restart")
    async def restart_session() -> dict[str, Any]:
        try:
            return await controller.restart()
        except GameStateError:
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    @app.post("/api/session/end")
    async def finish_study_session() -> dict[str, Any]:
        try:
            return await controller.finish_study()
        except GameStateError:
            raise
        except Exception as error:
            raise _public_http_error(error) from error

    return app


def _public_http_error(error: Exception) -> HTTPException:
    """Map runtime failures to useful messages without exposing credentials."""
    if isinstance(error, RemoteAuthenticationError):
        return HTTPException(status_code=401, detail="Среда отклонила ключ доступа.")
    if isinstance(error, RemoteAuthorizationError):
        return HTTPException(status_code=403, detail="Для этого ключа действие запрещено.")
    if isinstance(error, CompetitionClosedError):
        return HTTPException(status_code=409, detail="Выбранная арена сейчас закрыта.")
    if isinstance(error, RunCapacityExceededError):
        return HTTPException(
            status_code=409,
            detail="Нельзя создать run: уже есть активный запуск или исчерпана ёмкость.",
        )
    if isinstance(error, RemoteRateLimitError):
        return HTTPException(status_code=429, detail="Среда временно ограничила новые run.")
    if isinstance(error, RemoteRunNotFoundError):
        return HTTPException(
            status_code=410,
            detail="Run был остановлен средой соревнования и больше недоступен.",
        )
    if isinstance(error, RemoteRequestError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, RemoteUnavailableError):
        return HTTPException(status_code=503, detail="Рантайм BlueSec временно недоступен.")
    LOGGER.exception("Unhandled BlueSec web error", exc_info=error)
    return HTTPException(
        status_code=500,
        detail=f"Неожиданная ошибка локального сервера: {type(error).__name__}.",
    )


app = create_app()


def main() -> None:
    """Run a loopback-only development server."""
    port = int(os.getenv("BLUESEC_WEB_PORT", "8765"))
    uvicorn.run(
        "bluesec1_web.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
