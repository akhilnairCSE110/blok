// @ts-check
const messages = document.querySelector("#messages");
const form = document.querySelector("#form");
const prompt = document.querySelector("#prompt");
const tokens = document.querySelector("#tokens");
const send = document.querySelector("#send");
const clear = document.querySelector("#clear");
const key = "blok-chat-v1";
/** @type {{role: "user" | "assistant", text: string, meta?: string}[]} */
let history = [];

try { history = JSON.parse(localStorage.getItem(key) || "[]"); } catch { history = []; }

function render() {
  if (!messages) return;
  messages.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    const title = document.createElement("h1");
    title.textContent = "One model. Full context.";
    const text = document.createElement("p");
    text.textContent = "Prompt the pinned Kimi K2.6 runtime directly from this machine.";
    empty.append(title, text);
    messages.append(empty);
    return;
  }
  for (const item of history) {
    const row = document.createElement("article");
    row.className = `message ${item.role}`;
    const role = document.createElement("div");
    role.className = "role";
    role.textContent = item.role === "user" ? "You" : "Kimi";
    const content = document.createElement("div");
    content.className = "content";
    content.textContent = item.text;
    row.append(role, content);
    if (item.meta) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = item.meta;
      row.append(meta);
    }
    messages.append(row);
  }
  messages.lastElementChild?.scrollIntoView({behavior: "smooth", block: "end"});
}

function save() {
  localStorage.setItem(key, JSON.stringify(history.slice(-20)));
  render();
}

function conversation(next) {
  const prior = history.map(item => `${item.role === "user" ? "User" : "Assistant"}:\n${item.text}`).join("\n\n");
  return prior ? `Continue this conversation faithfully.\n\n${prior}\n\nUser:\n${next}` : next;
}

form?.addEventListener("submit", async event => {
  event.preventDefault();
  if (!(prompt instanceof HTMLTextAreaElement) || !(tokens instanceof HTMLInputElement) || !(send instanceof HTMLButtonElement)) return;
  const text = prompt.value.trim();
  const maxTokens = Number(tokens.value);
  if (!text || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 10000) return;
  const fullPrompt = conversation(text);
  history.push({role: "user", text});
  prompt.value = "";
  send.disabled = true;
  send.textContent = "Running…";
  save();
  const started = performance.now();
  try {
    const response = await fetch("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt: fullPrompt, max_tokens: maxTokens})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `request failed (${response.status})`);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    history.push({role: "assistant", text: body.text,
      meta: `${body.input_tokens.toLocaleString()} in · ${body.output_tokens.toLocaleString()} out · ${body.finish_reason} · ${seconds}s`});
  } catch (error) {
    history.push({role: "assistant", text: error instanceof Error ? error.message : String(error), meta: "error"});
  } finally {
    send.disabled = false;
    send.textContent = "Send";
    save();
    prompt.focus();
  }
});

prompt?.addEventListener("keydown", event => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) form?.requestSubmit();
});
clear?.addEventListener("click", () => { history = []; localStorage.removeItem(key); render(); });
render();
