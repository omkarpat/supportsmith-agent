// SupportSmith demo console — Phase 8.
//
// Vanilla JS, no build step. Drives the existing FastAPI endpoints:
//   POST /chat                              — send a message (mint or resume)
//   GET  /conversations?limit=50            — sidebar feed
//   GET  /conversations/{id}/messages       — full transcript for a thread
// Every request except /health carries `Authorization: Bearer <token>` so
// the Phase-7 bearer middleware on Railway accepts it.

// Namespace strings for browser storage slots — NOT secrets. The actual
// bearer token value is typed by the user into the input field at runtime
// and stashed in their browser only.
const TOKEN_KEY = "supportsmith.demo.token";
const REMEMBER_KEY = "supportsmith.demo.token.remember";
const SIDEBAR_LIMIT = 50;

const state = {
  token: "",
  remember: false,
  activeConversationId: null,
  conversations: [],
  sending: false,
};

const els = {
  app: document.querySelector(".app"),
  tokenInput: document.getElementById("token-input"),
  tokenRemember: document.getElementById("token-remember"),
  tokenClear: document.getElementById("token-clear"),
  newChat: document.getElementById("new-chat"),
  conversationList: document.getElementById("conversation-list"),
  sidebarStatus: document.getElementById("sidebar-status"),
  activeTitle: document.getElementById("active-title"),
  activeSubtitle: document.getElementById("active-subtitle"),
  messageList: document.getElementById("message-list"),
  emptyState: document.getElementById("empty-state"),
  banner: document.getElementById("banner"),
  composer: document.getElementById("composer"),
  composerInput: document.getElementById("composer-input"),
  composerSend: document.getElementById("composer-send"),
};

// --- token storage ----------------------------------------------------

function loadToken() {
  const remember = window.localStorage.getItem(REMEMBER_KEY) === "1";
  const store = remember ? window.localStorage : window.sessionStorage;
  const token = store.getItem(TOKEN_KEY) || "";
  state.token = token;
  state.remember = remember;
  els.tokenInput.value = token;
  els.tokenRemember.checked = remember;
}

function persistToken() {
  if (!state.token) {
    window.sessionStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(REMEMBER_KEY);
    return;
  }
  if (state.remember) {
    window.localStorage.setItem(TOKEN_KEY, state.token);
    window.localStorage.setItem(REMEMBER_KEY, "1");
    window.sessionStorage.removeItem(TOKEN_KEY);
  } else {
    window.sessionStorage.setItem(TOKEN_KEY, state.token);
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(REMEMBER_KEY);
  }
}

function clearToken() {
  state.token = "";
  els.tokenInput.value = "";
  persistToken();
  updateGatedControls();
}

// --- HTTP -------------------------------------------------------------

class TokenRequiredError extends Error {
  constructor(message = "Bearer token required") {
    super(message);
    this.name = "TokenRequiredError";
  }
}

async function apiFetch(path, { method = "GET", body, signal } = {}) {
  if (!state.token) {
    throw new TokenRequiredError();
  }
  const headers = { Authorization: `Bearer ${state.token}` };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
    credentials: "omit",
    cache: "no-store",
  });
  if (response.status === 401) {
    onUnauthorized();
    throw new TokenRequiredError("Token rejected (401)");
  }
  if (!response.ok) {
    const detail = await safeDetail(response);
    const err = new Error(detail || `${response.status} ${response.statusText}`);
    err.status = response.status;
    throw err;
  }
  if (response.status === 204) return null;
  return await response.json();
}

async function safeDetail(response) {
  try {
    const data = await response.json();
    if (data && typeof data.detail === "string") return data.detail;
    return JSON.stringify(data);
  } catch {
    return null;
  }
}

function onUnauthorized() {
  showBanner(
    "Token rejected. Paste a valid demo bearer token to continue.",
    "error",
  );
  els.tokenInput.focus();
  els.tokenInput.select();
}

// --- rendering --------------------------------------------------------

function showBanner(message, tone = "info") {
  els.banner.textContent = message;
  els.banner.dataset.tone = tone;
  els.banner.hidden = false;
}

function clearBanner() {
  els.banner.hidden = true;
  els.banner.textContent = "";
}

function setSidebarStatus(message, tone = "info") {
  if (!message) {
    els.sidebarStatus.hidden = true;
    els.sidebarStatus.textContent = "";
    return;
  }
  els.sidebarStatus.hidden = false;
  els.sidebarStatus.textContent = message;
  els.sidebarStatus.dataset.tone = tone;
}

function renderSidebar() {
  els.conversationList.replaceChildren();
  if (state.conversations.length === 0) {
    setSidebarStatus(
      state.token
        ? "No conversations yet — say hi below."
        : "Add a bearer token to load conversations.",
    );
    return;
  }
  setSidebarStatus("");
  const frag = document.createDocumentFragment();
  for (const conv of state.conversations) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conv-row";
    if (conv.conversation_id === state.activeConversationId) {
      button.setAttribute("aria-current", "true");
    }
    button.addEventListener("click", () =>
      loadConversation(conv.conversation_id),
    );

    const preview = document.createElement("div");
    preview.className = "conv-row__preview";
    preview.textContent =
      conv.last_message_preview ?? "(no messages yet)";
    const meta = document.createElement("div");
    meta.className = "conv-row__meta";
    const role = document.createElement("span");
    role.className = "conv-row__role";
    role.dataset.role = conv.last_role || "";
    role.textContent = conv.last_role
      ? `${conv.last_role} · turn ${conv.last_turn_number ?? "—"}`
      : "new";
    const stamp = document.createElement("span");
    stamp.textContent = formatRelative(conv.updated_at);
    meta.append(role, stamp);
    button.append(preview, meta);
    li.append(button);
    frag.append(li);
  }
  els.conversationList.append(frag);
}

function renderMessages(messages) {
  els.messageList.replaceChildren();
  if (!messages.length) {
    els.emptyState.hidden = false;
    return;
  }
  els.emptyState.hidden = true;
  const frag = document.createDocumentFragment();
  for (const msg of messages) {
    frag.append(buildMessageNode(msg));
  }
  els.messageList.append(frag);
  scrollMessagesToBottom();
}

function buildMessageNode(msg) {
  const li = document.createElement("li");
  li.className = `msg msg--${msg.role}`;
  if (msg.role !== "user") {
    const tag = document.createElement("span");
    tag.className = "msg__role-tag";
    tag.textContent =
      msg.role === "compliance" ? "compliance gate" : "agent";
    li.append(tag);
  }
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = msg.content;
  li.append(bubble);
  const metaNode = buildMetadataNode(msg);
  if (metaNode) li.append(metaNode);
  return li;
}

function appendOptimisticUser(text) {
  els.emptyState.hidden = true;
  const li = document.createElement("li");
  li.className = "msg msg--user msg--pending";
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = text;
  li.append(bubble);
  els.messageList.append(li);
  scrollMessagesToBottom();
  return li;
}

function appendTypingIndicator() {
  els.emptyState.hidden = true;
  const li = document.createElement("li");
  li.className = "msg msg--agent msg--typing";
  const tag = document.createElement("span");
  tag.className = "msg__role-tag";
  tag.textContent = "agent";
  li.append(tag);
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  for (let i = 0; i < 3; i += 1) {
    const dot = document.createElement("span");
    dot.className = "typing-dot";
    bubble.append(dot);
  }
  li.append(bubble);
  els.messageList.append(li);
  scrollMessagesToBottom();
  return li;
}

function buildMetadataNode(msg) {
  if (msg.role === "user") return null;
  const meta = msg.metadata || {};
  const fields = [];
  if (meta.source) fields.push(["source", meta.source]);
  if (typeof meta.verified === "boolean") {
    fields.push(["verified", meta.verified ? "yes" : "no"]);
  }
  if (typeof msg.turn_number === "number") {
    fields.push(["turn", String(msg.turn_number)]);
  }
  if (Array.isArray(meta.tools_used) && meta.tools_used.length) {
    fields.push(["tools used", meta.tools_used.join(", ")]);
  }
  if (
    Array.isArray(meta.matched_questions) &&
    meta.matched_questions.length
  ) {
    fields.push(["matched", meta.matched_questions.join(" · ")]);
  }
  if (!fields.length) return null;

  const wrap = document.createElement("div");
  wrap.className = "msg__meta";
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `details (${fields.length})`;
  details.append(summary);
  const grid = document.createElement("div");
  grid.className = "msg__meta-grid";
  for (const [key, value] of fields) {
    const k = document.createElement("span");
    k.className = "msg__meta-key";
    k.textContent = key;
    const v = document.createElement("span");
    v.className = "msg__meta-val";
    v.textContent = value;
    grid.append(k, v);
  }
  details.append(grid);
  wrap.append(details);
  return wrap;
}

function formatRelative(iso) {
  if (!iso) return "";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const diff = (Date.now() - then.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d`;
  return then.toLocaleDateString();
}

function scrollMessagesToBottom() {
  const container = els.messageList.parentElement;
  if (container) container.scrollTop = container.scrollHeight;
}

function setLoading(loading) {
  state.sending = loading;
  els.app.dataset.state = loading ? "loading" : "idle";
  updateGatedControls();
}

function updateGatedControls() {
  const enabled = Boolean(state.token);
  els.composerInput.disabled = !enabled || state.sending;
  els.composerSend.disabled = !enabled || state.sending;
  els.newChat.disabled = !enabled || state.sending;
  if (!enabled && !state.sending) {
    showBanner(
      "Add a bearer token (sidebar) to start chatting.",
      "info",
    );
  } else {
    clearBanner();
  }
}

function setActiveHeader(message) {
  if (state.activeConversationId) {
    els.activeTitle.textContent = `Conversation ${state.activeConversationId.slice(0, 8)}…`;
    els.activeSubtitle.textContent = message ?? "Resuming this thread.";
  } else {
    els.activeTitle.textContent = "New conversation";
    els.activeSubtitle.textContent =
      message ??
      "Start typing below — a fresh conversation is minted on the first message.";
  }
}

// --- actions ----------------------------------------------------------

async function refreshSidebar() {
  if (!state.token) {
    state.conversations = [];
    renderSidebar();
    return;
  }
  try {
    const data = await apiFetch(`/conversations?limit=${SIDEBAR_LIMIT}`);
    state.conversations = data?.conversations ?? [];
    renderSidebar();
  } catch (err) {
    if (err instanceof TokenRequiredError) return;
    setSidebarStatus(`Couldn't load conversations: ${err.message}`, "error");
  }
}

async function loadConversation(conversationId) {
  if (!state.token) return;
  state.activeConversationId = conversationId;
  setActiveHeader("Loading…");
  renderSidebar();
  try {
    const data = await apiFetch(
      `/conversations/${encodeURIComponent(conversationId)}/messages`,
    );
    renderMessages(data?.messages ?? []);
    setActiveHeader();
  } catch (err) {
    if (err instanceof TokenRequiredError) return;
    showBanner(`Couldn't load messages: ${err.message}`, "error");
  }
}

function startNewChat() {
  state.activeConversationId = null;
  renderMessages([]);
  renderSidebar();
  setActiveHeader();
  els.composerInput.focus();
}

async function sendMessage(text) {
  if (!text || state.sending) return;
  setLoading(true);
  appendOptimisticUser(text);
  const typingNode = appendTypingIndicator();
  try {
    const body = { message: text };
    if (state.activeConversationId) {
      body.conversation_id = state.activeConversationId;
    }
    const response = await apiFetch("/chat", { method: "POST", body });
    if (response?.conversation_id) {
      state.activeConversationId = response.conversation_id;
    }
    if (state.activeConversationId) {
      const data = await apiFetch(
        `/conversations/${encodeURIComponent(state.activeConversationId)}/messages`,
      );
      // ``renderMessages`` replaces the whole list, so the optimistic user
      // bubble and the typing indicator are swapped out for canonical rows
      // (with metadata) in one paint.
      renderMessages(data?.messages ?? []);
    }
    setActiveHeader();
    await refreshSidebar();
  } catch (err) {
    typingNode.remove();
    if (err instanceof TokenRequiredError) return;
    showBanner(`Chat failed: ${err.message}`, "error");
  } finally {
    setLoading(false);
  }
}

// --- event wiring -----------------------------------------------------

els.tokenInput.addEventListener("input", (event) => {
  state.token = event.target.value.trim();
  persistToken();
  updateGatedControls();
});

els.tokenInput.addEventListener("change", () => {
  if (state.token) {
    clearBanner();
    refreshSidebar();
  }
});

els.tokenRemember.addEventListener("change", (event) => {
  state.remember = event.target.checked;
  persistToken();
});

els.tokenClear.addEventListener("click", () => {
  clearToken();
  state.conversations = [];
  renderSidebar();
});

els.newChat.addEventListener("click", () => {
  startNewChat();
});

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = els.composerInput.value.trim();
  if (!text) return;
  els.composerInput.value = "";
  void sendMessage(text);
});

els.composerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

// --- boot -------------------------------------------------------------

loadToken();
updateGatedControls();
setActiveHeader();
renderMessages([]);
renderSidebar();
if (state.token) {
  void refreshSidebar();
}
