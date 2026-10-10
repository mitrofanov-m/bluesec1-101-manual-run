import asyncio
import json
from typing import Any

from bluesec1_web.mentor import (
    AdviceReply,
    ExplanationReply,
    MentorConfig,
    MentorService,
    evidence_call_fingerprint,
)


class CapturingMentorService(MentorService):
    def __init__(self) -> None:
        super().__init__(MentorConfig())
        self.prompt = ""

    async def _generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompt = prompt
        return {
            "meaning": "Инструмент вернул карточку компьютера.",
            "important": ["dns_hostname связывает хост с доменом."],
            "next_hint": "Сопоставьте хост с процессом.",
        }


def test_explanation_contains_only_decoded_selected_result() -> None:
    service = CapturingMentorService()
    result = {
        "id": "computer-01",
        "kind": "computer",
        "sam_account_name": "CRM86$",
        "dns_hostname": "CRM86.meridian.internal",
        "os": "Windows Server 2019",
    }
    event = {
        "kind": "tool_call",
        "tool_name": "get_ad_object_info",
        "arguments": {"ad_object_id": "computer-01"},
        "response": {
            "observation": {"last_tool_result": json.dumps(result)},
            "reward": 0,
            "terminated": False,
            "info": {"tool_status": "ok", "tool_result": result},
        },
    }
    context = {
        "phase": "active",
        "task_id": "task-01",
        "initial_observation": {"alert_text": "PowerShell started on CRM86"},
        "current_observation": {"padding": "x" * 60_000},
        "history": [{"padding": "y" * 60_000}],
    }

    reply = asyncio.run(service.explain(context, event))

    assert reply.meaning == "Инструмент вернул карточку компьютера."
    assert '"has_result": true' in service.prompt
    assert '"sam_account_name": "CRM86$"' in service.prompt
    assert "PowerShell started on CRM86" not in service.prompt
    assert '"task_id"' not in service.prompt
    assert '"padding"' not in service.prompt
    assert "…[данные сокращены локальным интерфейсом]" not in service.prompt


def test_explanation_marks_null_like_results_as_empty() -> None:
    service = CapturingMentorService()
    event = {
        "kind": "tool_call",
        "tool_name": "search",
        "response": {
            "observation": {"last_tool_result": "[]"},
            "info": {"tool_status": "ok", "tool_result": []},
        },
    }

    asyncio.run(service.explain({}, event))

    assert '"has_result": false' in service.prompt


def test_long_explanation_is_compacted_without_cutting_a_word() -> None:
    reply = ExplanationReply(
        meaning="Короткое значение.",
        important=[],
        next_hint="Проверьте сетевые соединения " + "очень " * 80,
    )

    assert len(reply.next_hint) <= 240
    assert reply.next_hint.endswith("…")
    assert not reply.next_hint.endswith("оче…")
    assert "maxLength" not in ExplanationReply.model_json_schema()["properties"]["next_hint"]


class SequencedMentorService(MentorService):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(MentorConfig())
        self.responses = responses
        self.prompts: list[str] = []

    async def _generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_repeated_completed_tool_call_is_retried_then_offers_close() -> None:
    duplicate = {
        "action": "call_tool",
        "explanation": "Ещё раз прочитаем ту же сущность.",
        "tool_name": "get_entity",
        "arguments_json": json.dumps(
            {"entity_id": "entity-01", "reasoning": "Перепишем обоснование."}
        ),
    }
    service = SequencedMentorService([duplicate.copy(), duplicate.copy()])
    context = {
        "conversation_history": [{"kind": "help", "tool_name": "get_entity"}],
        "completed_calls": [
            {
                "tool_name": "get_entity",
                "fingerprint": evidence_call_fingerprint(
                    "get_entity",
                    {"entity_id": "entity-01", "reasoning": "Первый просмотр."},
                ),
                "status": "ok",
            }
        ],
    }
    tools = [
        {
            "name": "get_entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["entity_id", "reasoning"],
            },
        }
    ]

    reply = asyncio.run(service.next_step(context, tools))

    assert reply.action == "ready_to_close"
    assert reply.tool_name is None
    assert len(service.prompts) == 2
    assert "в точности повторяет" in service.prompts[1]


def test_advice_keeps_conversation_history_when_runtime_evidence_is_large() -> None:
    service = SequencedMentorService(
        [
            {
                "title": "Новая мысль",
                "summary": "Смотрим на новый факт.",
                "details": [],
            }
        ]
    )
    context = {
        "conversation_history": [{"kind": "advice", "summary": "Уже проверили исходный процесс."}],
        "completed_calls": [],
        "latest_tool_call": {"result": {"large_result": "x" * 100_000}},
    }

    reply = asyncio.run(service.advice(context))

    assert isinstance(reply, AdviceReply)
    assert "Уже проверили исходный процесс." in service.prompts[0]
    evidence = service.prompts[0].split("EVIDENCE:\n", maxsplit=1)[1]
    assert json.loads(evidence)["context"]["conversation_history"][0]["kind"] == "advice"


def test_prepare_close_can_request_more_investigation() -> None:
    service = SequencedMentorService(
        [
            {
                "action": "need_investigation",
                "explanation": "Не проверена связь процесса с сетью.",
                "arguments_json": None,
            }
        ]
    )
    tool = {
        "name": "finish_investigation",
        "parameters": {"type": "object", "properties": {}},
    }

    reply = asyncio.run(service.prepare_close({"conversation_history": []}, tool))

    assert reply.action == "need_investigation"
    assert reply.arguments_json is None
    assert '"name": "finish_investigation"' in service.prompts[0]
