// Chat client: posts the question, reads the server-sent event stream and renders it live.

const $ = (id) => document.getElementById(id);
const messages = $("messages"), form = $("form"), input = $("input"), send = $("send");
let sessionId = null;
let busy = false;

const TOOL_LABELS = {
  search_regulation: "recherche",
  read_source: "lecture",
  lookup_definition: "définition",
};
const ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"];

// ---------- rendering -------------------------------------------------------------------------

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function sourceId(kind, number) {
  kind = kind.toLowerCase();
  if (kind.startsWith("art")) return `art-${parseInt(number, 10)}`;
  if (kind.startsWith("consid")) return `rct-${parseInt(number, 10)}`;
  return /^\d+$/.test(number) ? `anx-${ROMANS[number - 1]}` : `anx-${number.toUpperCase()}`;
}

// Small, safe Markdown subset: the text is escaped first, then a few patterns are re-enabled.
function renderMarkdown(text, statuses) {
  let html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[((?:Article|Annexe|Considérant)\s+(?:\d+|[IVX]+))[^\]]*\]/gi, (match, ref) => {
      const [kind, number] = ref.split(/\s+/);
      const id = sourceId(kind, number);
      const status = statuses ? statuses[id] || "" : "";
      const title = { verified: "Source lue par l'assistant", mentioned: "Référence citée dans un texte lu, non lue directement",
        unsupported: "Référence absente des passages lus : non vérifiée" }[status] || "";
      return `<button class="cite ${status}" data-source="${id}" title="${title}">${ref}</button>`;
    });
  const blocks = [];
  let list = null;
  for (const line of html.split("\n")) {
    const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (item) {
      if (!list) { list = []; blocks.push(list); }
      list.push(item[1]);
    } else {
      list = null;
      if (heading) blocks.push(`<h3>${heading[1]}</h3>`);
      else if (line.trim()) blocks.push(`<p>${line}</p>`);
    }
  }
  return blocks.map((b) => (Array.isArray(b) ? `<ul>${b.map((i) => `<li>${i}</li>`).join("")}</ul>` : b)).join("");
}

function addMessage(role) {
  $("welcome")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = role === "user" ? '<div class="bubble"></div>'
    : '<div class="bubble"><div class="steps"></div><div class="answer cursor"></div></div>';
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap;
}

function describeArguments(raw) {
  try { return Object.values(JSON.parse(raw)).join(" · "); } catch { return raw; }
}

function renderReport(final) {
  const c = final.citations;
  const parts = [];
  if (final.error) parts.push(`<span class="badge bad">erreur</span>`);
  else if (c.grounded) parts.push(`<span class="badge ok">✓ ${c.verified.length} source(s) vérifiée(s)</span>`);
  else if (c.unsupported.length) parts.push(`<span class="badge bad">⚠ ${c.unsupported.length} citation(s) non vérifiée(s)</span>`);
  else parts.push(`<span class="badge warn">réponse sans source</span>`);
  if (c.mentioned.length) parts.push(`<span class="badge warn">${c.mentioned.length} référence(s) indirecte(s)</span>`);
  parts.push(`${(final.latency_ms / 1000).toFixed(1)} s`);
  parts.push(`${final.steps.length} appel(s) d'outil`);
  const tokens = final.usage.prompt_tokens + final.usage.completion_tokens;
  if (tokens) parts.push(`${tokens.toLocaleString("fr-FR")} tokens`);
  if (final.usage.cost_eur) parts.push(`${final.usage.cost_eur.toFixed(4)} €`);
  parts.push(final.model);
  return `<div class="report">${parts.map((p) => (p.startsWith("<") ? p : `<span>${p}</span>`)).join("")}</div>`;
}

// ---------- streaming ---------------------------------------------------------------------------

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true; send.disabled = true;
  addMessage("user").querySelector(".bubble").textContent = question;
  const bot = addMessage("bot");
  const steps = bot.querySelector(".steps"), answer = bot.querySelector(".answer");
  let text = "";

  const handlers = {
    session: (d) => { sessionId = d.session_id; },
    tool_call: (d) => {
      const row = document.createElement("div");
      row.className = "step";
      row.innerHTML = `<code>${TOOL_LABELS[d.name] || escapeHtml(d.name)}</code><span class="args">${escapeHtml(describeArguments(d.arguments))}</span>`;
      steps.appendChild(row);
    },
    tool_result: (d) => {
      const row = steps.lastElementChild;
      if (row && d.labels.length) row.querySelector(".args").textContent += ` → ${d.labels.map((l) => l.label).join(", ")}`;
    },
    reset: () => { text = ""; answer.innerHTML = ""; },
    token: (d) => { text += d.text; answer.innerHTML = renderMarkdown(text); },
    error: (d) => { answer.insertAdjacentHTML("afterend", `<p class="error">${escapeHtml(d.message)}</p>`); },
    final: (d) => {
      const statuses = {};
      for (const status of ["verified", "mentioned", "unsupported"]) for (const id of d.citations[status]) statuses[id] = status;
      answer.classList.remove("cursor");
      answer.innerHTML = renderMarkdown(d.answer || text, statuses);
      answer.insertAdjacentHTML("afterend", renderReport(d));
    },
  };

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, session_id: sessionId }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
        const event = frame.match(/^event: (.*)$/m)?.[1], data = frame.match(/^data: (.*)$/m)?.[1];
        if (event && data && handlers[event]) handlers[event](JSON.parse(data));
        messages.scrollTop = messages.scrollHeight;
      }
    }
  } catch (err) {
    answer.insertAdjacentHTML("afterend", `<p class="error">Erreur : ${escapeHtml(String(err.message || err))}</p>`);
  } finally {
    answer.classList.remove("cursor");
    busy = false; send.disabled = false; input.focus();
  }
}

// ---------- source panel ------------------------------------------------------------------------

async function openSource(id) {
  const response = await fetch(`/api/sources/${encodeURIComponent(id)}`);
  if (!response.ok) return;
  const source = await response.json();
  $("panel-label").textContent = source.label;
  $("panel-title").textContent = source.title;
  $("panel-context").textContent = source.context;
  $("panel-text").textContent = source.text;
  $("panel").hidden = false;
}

// ---------- wiring ------------------------------------------------------------------------------

form.addEventListener("submit", (e) => { e.preventDefault(); const q = input.value; input.value = ""; input.style.height = ""; ask(q); });
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
input.addEventListener("input", () => { input.style.height = ""; input.style.height = `${input.scrollHeight}px`; });
messages.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite"), example = e.target.closest(".examples button");
  if (cite) openSource(cite.dataset.source);
  if (example) ask(example.textContent);
});
$("panel-close").addEventListener("click", () => { $("panel").hidden = true; });

fetch("/health").then((r) => r.json()).then((h) => {
  $("status").innerHTML = `${h.llm_model ? escapeHtml(h.llm_model) : "mode extractif (sans LLM)"}<br>${h.chunks} passages · recherche ${h.retrieval_mode}`;
}).catch(() => { $("status").textContent = "API injoignable"; });
