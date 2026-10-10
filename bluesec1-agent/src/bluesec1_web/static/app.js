"use strict";

let gameState = null;
let selectedToolName = null;
let renderedFormNode = null;

const elements = {
  connectView: document.querySelector("#connectView"),
  labView: document.querySelector("#labView"),
  connectForm: document.querySelector("#connectForm"),
  phaseBadge: document.querySelector("#phaseBadge"),
  toolCounter: document.querySelector("#toolCounter"),
  taskShortId: document.querySelector("#taskShortId"),
  currentJson: document.querySelector("#currentJson"),
  runMeta: document.querySelector("#runMeta"),
  timeline: document.querySelector("#timeline"),
  toolButtons: document.querySelector("#toolButtons"),
  toolFormArea: document.querySelector("#toolFormArea"),
  abortButton: document.querySelector("#abortButton"),
  restartButton: document.querySelector("#restartButton"),
  completionBanner: document.querySelector("#completionBanner"),
  busyOverlay: document.querySelector("#busyOverlay"),
  busyText: document.querySelector("#busyText"),
  toast: document.querySelector("#toast"),
};

const phaseLabels = {
  ready: "ГОТОВО",
  connecting: "ПОДКЛЮЧЕНИЕ",
  active: "РАССЛЕДОВАНИЕ",
  completed: "ЗАВЕРШЕНО",
  aborted: "ПРЕРВАНО",
  run_stopped: "RUN ОСТАНОВЛЕН",
  empty: "ОЧЕРЕДЬ ПУСТА",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error?.message || payload.detail || `HTTP ${response.status}`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return payload;
}

function setBusy(active, text = "Связь со средой…") {
  elements.busyText.textContent = text;
  elements.busyOverlay.classList.toggle("hidden", !active);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  window.setTimeout(() => elements.toast.classList.add("hidden"), 5500);
}

function parseMaybeJson(value) {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function pretty(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function decodeJsonContainer(value) {
  if (typeof value !== "string") return { value, encoded: false };
  const trimmed = value.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
    return { value, encoded: false };
  }
  try {
    const decoded = JSON.parse(trimmed);
    if (decoded && typeof decoded === "object") return { value: decoded, encoded: true };
  } catch {
    // A regular string may begin with a bracket; render it as a string in that case.
  }
  return { value, encoded: false };
}

function renderJsonTree(container, value) {
  container.replaceChildren(buildJsonNode(value, { key: null, depth: 0 }));
  container._copyValue = pretty(value);
}

function buildJsonNode(rawValue, { key, depth }) {
  const decoded = decodeJsonContainer(rawValue);
  const value = decoded.value;
  if (value && typeof value === "object") {
    const isArray = Array.isArray(value);
    const entries = isArray ? value.map((item, index) => [String(index), item]) : Object.entries(value);
    const details = document.createElement("details");
    details.className = "json-node";
    details.open = depth === 0 || (decoded.encoded && key === "alert_text");

    const summary = document.createElement("summary");
    if (key !== null) {
      const keyNode = document.createElement("span");
      keyNode.className = "json-key";
      keyNode.textContent = isArray && /^\d+$/.test(key) ? `[${key}]` : key;
      summary.append(keyNode, punctuation(":"));
    }
    const shape = document.createElement("span");
    shape.className = "json-shape";
    shape.textContent = isArray ? `[ ${entries.length} ]` : `{ ${entries.length} }`;
    summary.append(shape);
    if (decoded.encoded) {
      const badge = document.createElement("span");
      badge.className = "json-encoded-badge";
      badge.textContent = "JSON-строка";
      summary.append(badge);
    }

    const children = document.createElement("div");
    children.className = "json-children";
    entries.forEach(([childKey, childValue]) => {
      children.append(buildJsonNode(childValue, { key: childKey, depth: depth + 1 }));
    });
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "json-empty";
      empty.textContent = isArray ? "пустой массив" : "пустой объект";
      children.append(empty);
    }
    details.append(summary, children);
    return details;
  }

  const row = document.createElement("div");
  row.className = "json-leaf";
  if (key !== null) {
    const keyNode = document.createElement("span");
    keyNode.className = "json-key";
    keyNode.textContent = /^\d+$/.test(key) ? `[${key}]` : key;
    row.append(keyNode, punctuation(":"));
  }
  const scalar = document.createElement("span");
  scalar.className = `json-value json-${jsonScalarType(value)}`;
  scalar.textContent = formatJsonScalar(value);
  row.append(scalar);
  return row;
}

function punctuation(text) {
  const node = document.createElement("span");
  node.className = "json-punctuation";
  node.textContent = text;
  return node;
}

function jsonScalarType(value) {
  if (value === null) return "null";
  return typeof value;
}

function formatJsonScalar(value) {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  return String(value);
}

function shortId(value) {
  if (!value) return "—";
  return String(value).slice(0, 8).toUpperCase();
}

function updateState(state) {
  gameState = state;
  const phase = state.phase || "ready";
  elements.phaseBadge.textContent = phaseLabels[phase] || phase.toUpperCase();
  elements.phaseBadge.className = `phase-badge phase-${phase}`;
  elements.toolCounter.textContent = `${state.investigation_tool_calls || 0} действий`;

  const showLab = state.ever_started || state.task_id || !state.can_start;
  elements.connectView.classList.toggle("hidden", showLab);
  elements.labView.classList.toggle("hidden", !showLab);
  if (!showLab) return;

  renderCase(state);
  renderTimeline(state);
  renderTools(state);
  renderCompletion(state);
}

function renderCase(state) {
  elements.taskShortId.textContent = shortId(state.task_id);
  elements.runMeta.textContent = `RUN ${shortId(state.run_id)}\nTASK ${shortId(state.task_id)}`;
  renderJsonTree(elements.currentJson, state.current_observation);
  elements.abortButton.disabled = state.phase !== "active";
  elements.restartButton.disabled = state.phase === "connecting";
}

function renderEntryPoint(item, kind) {
  const card = document.createElement("div");
  card.className = "entry-point";
  const type = document.createElement("span");
  type.className = "entity-type";
  type.textContent = `${kind} // ${item.type || "unknown"}`;
  const name = document.createElement("span");
  name.className = "entity-name";
  name.textContent =
    item.properties?.hostname ||
    item.properties?.image_path ||
    item.properties?.name ||
    item.id ||
    "Без имени";
  const id = document.createElement("span");
  id.className = "entity-id";
  id.textContent = item.id || "id отсутствует";
  card.append(type, name, id);
  return card;
}

function renderTimeline(state) {
  const events = (state.history || []).filter((event) => event.kind !== "task_started");
  elements.timeline.replaceChildren(renderInitialAlertEvent(state));
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "empty-timeline";
    empty.textContent = "Теперь выберите первый инструмент справа.";
    elements.timeline.append(empty);
    return;
  }

  events.forEach((event, index) => {
    const article = document.createElement("article");
    article.className = `timeline-event${event.error ? " event-error" : ""}`;
    const header = document.createElement("div");
    header.className = "event-header";
    const tool = document.createElement("span");
    tool.className = "event-tool";
    tool.textContent =
      event.kind === "task_aborted"
        ? "ABORT TASK"
        : event.kind === "run_stopped"
          ? "RUN STOPPED"
        : `${String(index + 1).padStart(2, "0")} / ${event.tool_name || event.kind}`;
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = event.at ? new Date(event.at).toLocaleTimeString("ru-RU") : "";
    header.append(tool, time);

    const body = document.createElement("div");
    body.className = "event-body";
    if (event.response) {
      const tags = document.createElement("div");
      tags.className = "response-summary";
      const response = event.response;
      [
        `status: ${response.info?.tool_status || (response.terminated ? "terminal" : "received")}`,
        `reward: ${response.reward ?? 0}`,
        `terminated: ${Boolean(response.terminated)}`,
      ].forEach((label) => {
        const tag = document.createElement("span");
        tag.className = "response-tag";
        tag.textContent = label;
        tags.append(tag);
      });
      body.append(tags);
    }
    if (event.arguments) body.append(renderCodeSection("ОТПРАВЛЕНО", event.arguments));
    if (event.response) body.append(renderCodeSection("ПОЛНЫЙ ОТВЕТ СРЕДЫ", event.response));
    if (event.task_result) body.append(renderCodeSection("РЕЗУЛЬТАТ", event.task_result));
    if (event.error) body.append(renderCodeSection("ОШИБКА", event.error));
    article.append(header, body);
    elements.timeline.append(article);
  });
}

function renderInitialAlertEvent(state) {
  const observation = state.initial_observation || {};
  const parsedAlert = parseMaybeJson(observation.alert_text);
  const alert = parsedAlert && typeof parsedAlert === "object" ? parsedAlert : {};
  const entities = Array.isArray(alert.trigger_entities) ? alert.trigger_entities : [];
  const relations = Array.isArray(alert.trigger_relations) ? alert.trigger_relations : [];
  const description =
    alert.alert_description ||
    (typeof parsedAlert === "string" ? parsedAlert : "Среда не передала alert_description.");
  const startedEvent = (state.history || []).find((event) => event.kind === "task_started");

  const article = document.createElement("article");
  article.className = "timeline-event initial-alert-event";

  const header = document.createElement("div");
  header.className = "event-header";
  const title = document.createElement("span");
  title.className = "event-tool";
  title.textContent = "00 / ИСХОДНЫЙ АЛЕРТ";
  const time = document.createElement("span");
  time.className = "event-time";
  time.textContent = startedEvent?.at
    ? new Date(startedEvent.at).toLocaleTimeString("ru-RU")
    : "СТАРТ ЗАДАЧИ";
  header.append(title, time);

  const body = document.createElement("div");
  body.className = "event-body";
  const label = document.createElement("h4");
  label.textContent = "УСЛОВИЕ ЗАДАЧИ";
  const text = document.createElement("p");
  text.className = "initial-alert-description";
  text.textContent = description;
  body.append(label, text, renderInitialEntryPoints(entities, relations));

  const rawDetails = document.createElement("details");
  rawDetails.className = "raw-details initial-raw-details";
  const rawSummary = document.createElement("summary");
  rawSummary.textContent = "Все исходные поля среды";
  const toolbar = document.createElement("div");
  toolbar.className = "json-toolbar";
  const toolbarLabel = document.createElement("span");
  toolbarLabel.textContent = "initial_observation";
  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.dataset.copyTarget = "initialJson";
  copyButton.textContent = "Копировать";
  toolbar.append(toolbarLabel, copyButton);
  const initialJson = document.createElement("div");
  initialJson.id = "initialJson";
  initialJson.className = "json-tree";
  initialJson.setAttribute("aria-label", "Исходный JSON среды");
  renderJsonTree(initialJson, state.initial_observation);
  rawDetails.append(rawSummary, toolbar, initialJson);
  body.append(rawDetails);

  article.append(header, body);
  return article;
}

function renderInitialEntryPoints(entities, relations) {
  const section = document.createElement("section");
  section.className = "initial-entry-section";
  const heading = document.createElement("div");
  heading.className = "section-title";
  const label = document.createElement("span");
  label.textContent = "Точки входа";
  const count = document.createElement("span");
  count.className = "count";
  count.textContent = String(entities.length + relations.length);
  heading.append(label, count);

  const list = document.createElement("div");
  list.className = "entity-list initial-entity-list";
  list.append(
    ...entities.map((entity) => renderEntryPoint(entity, "entity")),
    ...relations.map((relation) => renderEntryPoint(relation, "relation")),
  );
  if (!entities.length && !relations.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "Trigger entities и relations отсутствуют.";
    list.append(empty);
  }
  section.append(heading, list);
  return section;
}

function renderCodeSection(title, value) {
  const container = document.createElement("section");
  const heading = document.createElement("h4");
  heading.textContent = title;
  container.append(heading);
  const decoded = decodeJsonContainer(value);
  if (decoded.encoded || (value !== null && typeof value === "object")) {
    const tree = document.createElement("div");
    tree.className = "json-tree event-json-tree";
    renderJsonTree(tree, value);
    container.append(tree);
  } else {
    const pre = document.createElement("pre");
    pre.textContent = typeof value === "string" ? value : pretty(value);
    container.append(pre);
  }
  return container;
}

function renderCompletion(state) {
  const result = state.task_result;
  const terminal = ["completed", "aborted", "run_stopped", "empty"].includes(state.phase);
  elements.completionBanner.classList.toggle("hidden", !terminal);
  if (!terminal) return;
  elements.completionBanner.replaceChildren();
  elements.completionBanner.classList.toggle("run-stopped-banner", state.phase === "run_stopped");
  const heading = document.createElement("h3");
  heading.textContent =
    state.phase === "completed"
      ? "Расследование завершено"
      : state.phase === "run_stopped"
        ? "Run остановлен средой соревнования"
        : "Сессия закрыта";
  const copy = document.createElement("p");
  copy.textContent =
    state.phase === "completed"
      ? "Среда приняла терминальный вызов. Вторая задача запрошена не будет."
      : state.phase === "run_stopped"
        ? "Продолжить этот run невозможно. Хотите начать новый run?"
      : state.phase === "empty"
        ? "Для этого run среда не вернула задачу."
        : "Задача прервана без финального вердикта. Вторая задача запрошена не будет.";
  elements.completionBanner.append(heading, copy);
  if (state.phase === "run_stopped" && state.can_restart) {
    const restart = document.createElement("button");
    restart.type = "button";
    restart.className = "restart-button";
    restart.textContent = "Начать новый run";
    restart.addEventListener("click", () => restartGame());
    elements.completionBanner.append(restart);
  }
  if (result) {
    const stats = document.createElement("p");
    stats.textContent = `reason=${result.completion_reason} · score=${result.total_reward} · calls=${result.tool_calls}`;
    elements.completionBanner.append(stats);
  }
}

function renderTools(state) {
  const tools = state.available_tools || [];
  const isActive = state.phase === "active";
  elements.toolButtons.replaceChildren();
  tools.forEach((tool) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tool-button";
    if (tool.name === "finish_investigation") button.classList.add("finish-tool");
    if (tool.name === selectedToolName) button.classList.add("active");
    button.disabled = !isActive;
    button.textContent = tool.name;
    button.title = tool.description || tool.name;
    button.addEventListener("click", () => {
      selectedToolName = tool.name;
      renderTools(gameState);
    });
    elements.toolButtons.append(button);
  });

  if (!isActive) {
    selectedToolName = null;
    elements.toolFormArea.className = "tool-form-area empty-tool-state";
    elements.toolFormArea.innerHTML = `<div class="crosshair" aria-hidden="true">×</div><p>${
      state.phase === "run_stopped"
        ? "Run остановлен средой. Продолжение недоступно."
        : "Сессия закрыта. Новые действия недоступны."
    }</p>`;
  } else if (selectedToolName) {
    const selected = tools.find((tool) => tool.name === selectedToolName);
    if (selected) renderToolForm(selected);
  }
}

function renderToolForm(tool) {
  elements.toolFormArea.className = "tool-form-area";
  elements.toolFormArea.replaceChildren();
  const title = document.createElement("div");
  title.className = "tool-form-title";
  const heading = document.createElement("h3");
  heading.textContent = tool.name;
  const description = document.createElement("p");
  description.textContent = tool.description || "Параметры предоставлены средой BlueSec.";
  title.append(heading, description);

  const form = document.createElement("form");
  form.noValidate = true;
  const schema = tool.parameters || { type: "object", properties: {} };
  renderedFormNode = buildSchemaNode(schema, schema, {
    label: null,
    required: true,
    path: tool.name,
  });
  form.append(renderedFormNode);
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "tool-submit";
  if (tool.name === "finish_investigation") submit.classList.add("finish-submit");
  submit.innerHTML = `<span>${
    tool.name === "finish_investigation" ? "Отправить финальный вердикт" : "Выполнить действие"
  }</span><span aria-hidden="true">→</span>`;
  form.append(submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const argumentsPayload = collectNode(renderedFormNode, true);
      if (
        tool.name === "finish_investigation" &&
        !window.confirm(
          "Это завершит первую задачу и закроет run. После отправки расследование продолжить нельзя. Отправить вердикт?",
        )
      ) {
        return;
      }
      setBusy(true, `Выполняется ${tool.name}…`);
      const state = await api("/api/tool", {
        method: "POST",
        body: JSON.stringify({ tool_name: tool.name, arguments: argumentsPayload }),
      });
      selectedToolName = null;
      updateState(state);
    } catch (error) {
      showToast(error.message);
    } finally {
      setBusy(false);
    }
  });
  elements.toolFormArea.append(title, form);
}

function resolveSchema(schema, root) {
  let resolved = schema || {};
  const seen = new Set();
  while (resolved.$ref && !seen.has(resolved.$ref)) {
    seen.add(resolved.$ref);
    const parts = resolved.$ref.replace(/^#\//, "").split("/");
    let target = root;
    for (const part of parts) target = target?.[part.replace(/~1/g, "/").replace(/~0/g, "~")];
    resolved = target || resolved;
  }
  if (Array.isArray(resolved.allOf)) {
    return resolved.allOf.reduce((acc, part) => ({ ...acc, ...resolveSchema(part, root) }), {
      ...resolved,
      allOf: undefined,
    });
  }
  return resolved;
}

function schemaLabel(schema, fallback, index) {
  const resolved = schema || {};
  const discriminator = Object.values(resolved.properties || {}).find(
    (property) => property && Object.hasOwn(property, "const"),
  );
  return resolved.title || discriminator?.const || fallback || `Вариант ${index + 1}`;
}

function buildSchemaNode(schema, root, options) {
  const resolved = resolveSchema(schema, root);
  const variants = (resolved.oneOf || resolved.anyOf || []).filter(
    (variant) => resolveSchema(variant, root).type !== "null",
  );
  if (variants.length === 1) {
    return buildSchemaNode(
      { ...resolveSchema(variants[0], root), description: resolved.description },
      root,
      options,
    );
  }
  if (Array.isArray(variants) && variants.length) {
    return buildVariantNode(resolved, variants, root, options);
  }

  const type = Array.isArray(resolved.type)
    ? resolved.type.find((value) => value !== "null")
    : resolved.type;
  if (type === "object" || resolved.properties) return buildObjectNode(resolved, root, options);
  if (type === "array") return buildArrayNode(resolved, root, options);
  return buildPrimitiveNode(resolved, root, options);
}

function fieldShell(label, required, description) {
  const field = document.createElement("div");
  field.className = "schema-field";
  if (label) {
    const labelNode = document.createElement("label");
    labelNode.className = "field-label";
    labelNode.textContent = label;
    if (required) {
      const mark = document.createElement("span");
      mark.className = "required-mark";
      mark.textContent = "*";
      labelNode.append(mark);
    }
    field.append(labelNode);
  }
  if (description) {
    const copy = document.createElement("p");
    copy.className = "field-description";
    copy.textContent = description;
    field.append(copy);
  }
  return field;
}

function buildVariantNode(schema, variants, root, options) {
  const wrapper = fieldShell(options.label, options.required, schema.description);
  wrapper.dataset.kind = "variant";
  wrapper.dataset.required = String(Boolean(options.required));
  const contentWrapper = document.createElement("div");
  if (!options.required) {
    wrapper._enabled = addOptionalToggle(wrapper, contentWrapper);
  }
  const select = document.createElement("select");
  select.className = "variant-select";
  variants.forEach((variant, index) => {
    const resolvedVariant = resolveSchema(variant, root);
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = schemaLabel(resolvedVariant, null, index);
    select.append(option);
  });
  const content = document.createElement("div");
  const renderVariant = () => {
    const selectedSchema = resolveSchema(variants[Number(select.value)], root);
    wrapper._activeNode = buildSchemaNode(selectedSchema, root, {
      label: null,
      required: true,
      path: `${options.path}.variant`,
    });
    content.replaceChildren(wrapper._activeNode);
  };
  select.addEventListener("change", renderVariant);
  contentWrapper.append(select, content);
  wrapper.append(contentWrapper);
  renderVariant();
  return wrapper;
}

function buildObjectNode(schema, root, options) {
  const wrapper = fieldShell(options.label, options.required, schema.description);
  wrapper.dataset.kind = "object";
  wrapper.dataset.required = String(Boolean(options.required));
  if (options.label) wrapper.classList.add("nested-object");
  wrapper._fields = [];
  const content = document.createElement("div");
  if (!options.required) {
    wrapper._enabled = addOptionalToggle(wrapper, content);
  }
  const required = new Set(schema.required || []);
  Object.entries(schema.properties || {}).forEach(([name, propertySchema]) => {
    const child = buildSchemaNode(propertySchema, root, {
      label: propertySchema.title || name,
      required: required.has(name),
      path: `${options.path}.${name}`,
    });
    wrapper._fields.push({ name, node: child, required: required.has(name) });
    content.append(child);
  });
  wrapper.append(content);
  return wrapper;
}

function addOptionalToggle(wrapper, content) {
  const toggleLabel = document.createElement("label");
  toggleLabel.className = "optional-toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  const text = document.createElement("span");
  text.textContent = "Задать этот необязательный параметр";
  toggleLabel.append(checkbox, text);
  content.classList.add("optional-content", "hidden");
  checkbox.addEventListener("change", () => {
    content.classList.toggle("hidden", !checkbox.checked);
  });
  wrapper.append(toggleLabel);
  return checkbox;
}

function buildArrayNode(schema, root, options) {
  const wrapper = fieldShell(options.label, options.required, schema.description);
  wrapper.dataset.kind = "array";
  wrapper.dataset.required = String(Boolean(options.required));
  wrapper.dataset.minItems = String(Number(schema.minItems || 0));
  wrapper._items = [];
  const box = document.createElement("div");
  box.className = "schema-array";
  const list = document.createElement("div");
  const add = document.createElement("button");
  add.type = "button";
  add.className = "array-add";
  add.textContent = "+ Добавить элемент";
  const addItem = () => {
    const row = document.createElement("div");
    row.className = "array-item";
    const child = buildSchemaNode(schema.items || { type: "string" }, root, {
      label: null,
      required: true,
      path: `${options.path}[${wrapper._items.length}]`,
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "array-item-remove";
    remove.textContent = "×";
    remove.title = "Удалить элемент";
    const item = { row, node: child };
    remove.addEventListener("click", () => {
      wrapper._items = wrapper._items.filter((candidate) => candidate !== item);
      row.remove();
    });
    row.append(remove, child);
    wrapper._items.push(item);
    list.append(row);
  };
  add.addEventListener("click", addItem);
  box.append(list, add);
  wrapper.append(box);
  const initialItems = Number(schema.minItems || 0);
  for (let index = 0; index < initialItems; index += 1) addItem();
  return wrapper;
}

function buildPrimitiveNode(schema, _root, options) {
  const wrapper = fieldShell(options.label, options.required, schema.description);
  wrapper.dataset.required = String(Boolean(options.required));
  if (Object.hasOwn(schema, "const")) {
    wrapper.dataset.kind = "const";
    wrapper._value = schema.const;
    const display = document.createElement("div");
    display.className = "constant-value";
    display.textContent = String(schema.const);
    wrapper.append(display);
    return wrapper;
  }

  const enumValues = schema.enum;
  if (Array.isArray(enumValues)) {
    wrapper.dataset.kind = "enum";
    const select = document.createElement("select");
    if (!options.required) {
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "— не задано —";
      select.append(blank);
    }
    enumValues.forEach((value) => {
      const option = document.createElement("option");
      option.value = JSON.stringify(value);
      option.textContent = String(value);
      if (value === schema.default) option.selected = true;
      select.append(option);
    });
    wrapper._input = select;
    wrapper.append(select);
    return wrapper;
  }

  const type = schema.type || "string";
  wrapper.dataset.kind = type;
  let input;
  if (type === "boolean") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(schema.default);
  } else if (type === "integer" || type === "number") {
    input = document.createElement("input");
    input.type = "number";
    input.step = type === "integer" ? "1" : "any";
    if (schema.minimum !== undefined) input.min = String(schema.minimum);
    if (schema.maximum !== undefined) input.max = String(schema.maximum);
    if (schema.default !== undefined) input.value = String(schema.default);
  } else if (options.path.toLowerCase().includes("reasoning")) {
    input = document.createElement("textarea");
    input.placeholder = "Обоснуйте, зачем нужен этот шаг…";
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.placeholder = schema.examples?.[0] || schema.default || "";
    if (schema.default !== undefined) input.value = String(schema.default);
  }
  input.required = Boolean(options.required);
  wrapper._input = input;
  wrapper.append(input);

  if (/(entity_id|relation_id|ad_object_id)$/i.test(options.path)) {
    const fieldName = options.path.split(".").pop();
    const suggestions = knownIds(fieldName);
    if (suggestions.length) {
      const box = document.createElement("div");
      box.className = "id-suggestions";
      suggestions.slice(0, 8).forEach((suggestion) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = suggestion.id;
        button.title = suggestion.label;
        button.dataset.tooltip = suggestion.label;
        button.setAttribute(
          "aria-label",
          `${suggestion.label}. Подставить ID ${suggestion.id}`,
        );
        button.addEventListener("click", () => {
          input.value = suggestion.id;
          input.focus();
        });
        box.append(button);
      });
      wrapper.append(box);
    }
  }
  return wrapper;
}

function collectNode(node, forceRequired = false) {
  const required = forceRequired || node.dataset.required === "true";
  const kind = node.dataset.kind;
  if (node._enabled && !node._enabled.checked) return undefined;
  if (kind === "variant") return collectNode(node._activeNode, required);
  if (kind === "object") {
    const value = {};
    for (const field of node._fields || []) {
      const nested = collectNode(field.node, field.required);
      if (nested !== undefined) value[field.name] = nested;
    }
    if (!Object.keys(value).length && !required) return undefined;
    return value;
  }
  if (kind === "array") {
    const values = (node._items || []).map((item) => collectNode(item.node, true));
    if (!values.length && !required) return undefined;
    const minItems = Number(node.dataset.minItems || 0);
    if (values.length < minItems) {
      throw new Error(`Добавьте минимум ${minItems} элементов.`);
    }
    return values;
  }
  if (kind === "const") return node._value;
  if (kind === "enum") {
    if (!node._input.value) {
      if (required) throw new Error("Выберите значение обязательного поля.");
      return undefined;
    }
    return JSON.parse(node._input.value);
  }
  if (kind === "boolean") return Boolean(node._input.checked);
  const raw = node._input?.value?.trim() ?? "";
  if (!raw) {
    if (required) {
      const label = node.querySelector(".field-label")?.textContent || "Поле";
      throw new Error(`${label.replace("*", "")} обязательно.`);
    }
    return undefined;
  }
  if (kind === "integer") return Number.parseInt(raw, 10);
  if (kind === "number") return Number.parseFloat(raw);
  return raw;
}

function knownIds(requestedField = "entity_id") {
  const values = {
    entity_id: new Map(),
    relation_id: new Map(),
    ad_object_id: new Map(),
  };
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f-]{27,}$/i;
  const add = (bucket, value, label = null) => {
    if (typeof value === "string" && value.length <= 128 && uuidPattern.test(value)) {
      const fallbackType = bucket.replace(/_id$/, "");
      const suggestion = {
        id: value,
        label: label || `${fallbackType}: ${value}`,
      };
      if (label || !values[bucket].has(value)) values[bucket].set(value, suggestion);
    }
  };
  const visit = (value, key = "", parent = null) => {
    if (typeof value === "string") {
      const decoded = parseMaybeJson(value);
      if (decoded !== value) visit(decoded, key);
      if (key === "entity_id") add("entity_id", value, describeKnownObject(parent, "entity"));
      if (key === "relation_id") {
        add("relation_id", value, describeKnownObject(parent, "relation"));
      }
      if (key === "ad_object_id") {
        add("ad_object_id", value, describeKnownObject(parent, "ad_object"));
      }
      if (key === "source_id" || key === "target_id" || key === "host_id") {
        add("entity_id", value);
      }
      if (key === "id" && parent) {
        if (Object.hasOwn(parent, "source_id") && Object.hasOwn(parent, "target_id")) {
          add("relation_id", value, describeKnownObject(parent, "relation"));
        } else {
          add("entity_id", value, describeKnownObject(parent, "entity"));
        }
      }
      return;
    }
    if (Array.isArray(value)) value.forEach((item) => visit(item, key));
    else if (value && typeof value === "object") {
      Object.entries(value).forEach(([nestedKey, nestedValue]) =>
        visit(nestedValue, nestedKey, value),
      );
    }
  };
  visit(gameState?.initial_observation);
  visit(gameState?.current_observation);
  visit(gameState?.history);
  Object.values(values).forEach((bucket) => {
    bucket.delete(gameState?.run_id);
    bucket.delete(gameState?.task_id);
  });
  return [...(values[requestedField] || values.entity_id).values()];
}

function describeKnownObject(value, fallbackType) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const properties =
    value.properties && typeof value.properties === "object" ? value.properties : {};
  const type = String(value.type || fallbackType || "object");
  let displayValue;

  if (type === "windows_process") {
    displayValue =
      properties.cmdline ||
      properties.image_path ||
      properties.original_filename ||
      properties.name;
  } else if (type === "host") {
    displayValue = properties.hostname || properties.name || properties.fqdn;
  } else if (fallbackType === "relation") {
    displayValue =
      properties.name ||
      (value.source_id && value.target_id ? `${value.source_id} → ${value.target_id}` : null);
  } else {
    displayValue =
      properties.display_name ||
      properties.name ||
      properties.hostname ||
      properties.cmdline ||
      properties.image_path ||
      properties.user ||
      properties.sam_account_name;
  }

  return displayValue ? `${type}: ${displayValue}` : null;
}

elements.connectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(elements.connectForm);
  const token = String(data.get("token") || "");
  setBusy(true, "Создаём run и получаем первую задачу…");
  try {
    const state = await api("/api/session/start", {
      method: "POST",
      body: JSON.stringify({
        endpoint: String(data.get("endpoint") || "bluesec.team:443"),
        token,
        arena: String(data.get("arena") || "").trim() || null,
        agent_name: String(data.get("agentName") || "bluesec-manual-lab"),
      }),
    });
    elements.connectForm.elements.token.value = "";
    updateState(state);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
});

elements.abortButton.addEventListener("click", async () => {
  if (!window.confirm("Прервать первую задачу без вердикта и закрыть run?")) return;
  setBusy(true, "Прерываем задачу и закрываем run…");
  try {
    updateState(await api("/api/session/abort", { method: "POST", body: "{}" }));
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
});

async function restartGame({ confirmActive = false } = {}) {
  if (
    confirmActive &&
    gameState?.phase === "active" &&
    !window.confirm(
      "Текущая задача будет прервана без вердикта. После этого откроется форма нового run. Начать заново?",
    )
  ) {
    return;
  }
  setBusy(true, "Подготавливаем новую игру…");
  try {
    updateState(await api("/api/session/restart", { method: "POST", body: "{}" }));
    elements.connectForm.elements.token.focus();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

elements.restartButton.addEventListener("click", () => restartGame({ confirmActive: true }));

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-target]");
  if (!button) return;
  const target = document.querySelector(`#${button.dataset.copyTarget}`);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target._copyValue ?? target.textContent);
    button.textContent = "Скопировано";
    window.setTimeout(() => (button.textContent = "Копировать"), 1200);
  } catch {
    showToast("Не удалось скопировать данные.");
  }
});

api("/api/state")
  .then(updateState)
  .catch((error) => showToast(`Локальный API недоступен: ${error.message}`));
