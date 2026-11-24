import { pcm16Wav } from "./audio-engine.mjs?v=call-history";

function encodeWav(wav) {
  if (!wav) return null;
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",", 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(wav);
  });
}

async function saveTurn(entry) {
  const button = entry.row.querySelector(".history-save");
  const status = entry.row.querySelector(".history-save-status");
  button.disabled = true;
  status.textContent = "正在保存…";
  try {
    entry.saveId ??= crypto.randomUUID();
    const [questionWav, replyWav] = await Promise.all([encodeWav(entry.questionWav), encodeWav(entry.replyWav)]);
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: entry.saveId,
        turn: entry.id,
        status: entry.status.textContent,
        input_meta: entry.inputMeta || "",
        reply: entry.text,
        retrieval: entry.retrieval ?? null,
        question_wav: questionWav,
        reply_wav: replyWav,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      status.textContent = error?.detail || "保存失败，请重试。";
      button.disabled = false;
      return;
    }
    status.textContent = "已保存到数据飞轮";
    button.hidden = true;
  } catch (_) {
    status.textContent = "保存失败，请重试。";
    button.disabled = false;
  }
}

function setPreview(player, wav) {
  const previous = player.getAttribute("src");
  player.pause();
  if (previous) URL.revokeObjectURL(previous);
  if (wav) player.src = URL.createObjectURL(wav);
  else player.removeAttribute("src");
  player.hidden = !wav;
  player.load();
}

function renderRetrieval(results, retrieval) {
  const selected = retrieval?.selected_metric;
  const matches = retrieval?.[selected];
  results.replaceChildren();
  if (!matches?.length) return 0;
  const labels = { euclidean: "平方欧氏距离（越小越靠前）", cosine: "余弦相似度（越大越靠前）" };
  const other = selected === "euclidean" ? "cosine" : "euclidean";
  for (const metric of [selected, other]) {
    if (!retrieval[metric]?.length) continue;
    const heading = document.createElement("h3");
    heading.textContent = `${metric === selected ? "用于生成回答" : "对照排序"} · ${labels[metric]}`;
    const list = document.createElement("ol");
    for (const hit of retrieval[metric]) {
      const item = document.createElement("li");
      const score = document.createElement("strong");
      score.textContent = Number.isFinite(hit.score) ? `分数 ${hit.score.toPrecision(6)}` : "分数不可用";
      const passage = document.createElement("div");
      passage.textContent = hit.passage;
      item.append(score, passage);
      list.append(item);
    }
    results.append(heading, list);
  }
  return matches.length;
}

function renderLinks(panel, links) {
  const list = panel.lastElementChild;
  list.replaceChildren();
  for (const address of links) {
    let url;
    try { url = new URL(address); } catch (_) { continue; }
    if (url.protocol !== "https:" && url.protocol !== "http:") continue;
    const anchor = document.createElement("a");
    anchor.href = url.href;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.textContent = url.host + (url.pathname === "/" ? "" : url.pathname);
    const item = document.createElement("li");
    item.append(anchor);
    list.append(item);
  }
  panel.hidden = !list.childElementCount;
}

export class CallHistory {
  constructor(registerPreview) {
    this.scroll = document.getElementById("history-scroll");
    this.list = document.getElementById("history-turns");
    this.empty = document.getElementById("history-empty");
    this.template = document.getElementById("history-turn-template");
    this.registerPreview = registerPreview;
    this.current = null;
  }

  clear() {
    for (const player of this.list.querySelectorAll("audio[src]")) setPreview(player, null);
    this.list.replaceChildren();
    this.empty.hidden = false;
    this.current = null;
  }

  follow(update) {
    const { scrollTop, scrollHeight, clientHeight } = this.scroll;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 48;
    update();
    if (nearBottom) this.scroll.scrollTop = this.scroll.scrollHeight;
  }

  begin(id, sampleRate) {
    this.finish({ status: "回复已中断" });
    const row = this.template.content.firstElementChild.cloneNode(true);
    row.querySelector(".history-turn-heading").textContent = `第 ${id} 轮`;
    const status = row.querySelector(".history-status");
    status.textContent = "等待回复";
    const input = row.querySelector(".history-input");
    const output = row.querySelector(".history-output");
    input.setAttribute("aria-label", `第 ${id} 轮您的录音`);
    this.registerPreview(input);
    this.registerPreview(output);
    const entry = { id, sampleRate, row, status, input, output, chunks: [], finished: false };
    row.querySelector(".history-save").addEventListener("click", () => saveTurn(entry));
    this.current = entry;
    this.follow(() => {
      this.empty.hidden = true;
      this.list.append(row);
    });
  }

  setInput(id, wav, meta) {
    const entry = this.current;
    if (entry?.id !== id || entry.finished) return;
    entry.questionWav = wav;
    entry.inputMeta = meta;
    this.follow(() => {
      const label = entry.row.querySelector(".history-input-meta");
      label.textContent = meta;
      label.hidden = false;
      setPreview(entry.input, wav);
    });
  }

  setRetrieval(retrieval, links = []) {
    const entry = this.current;
    if (!entry || entry.finished || !retrieval || entry.retrieval) return;
    entry.retrieval = retrieval;
    this.follow(() => {
      const panel = entry.row.querySelector(".history-retrieval");
      const count = renderRetrieval(panel.lastElementChild, retrieval);
      panel.hidden = !count;
      if (count) panel.firstElementChild.textContent = `检索结果 · Top-${count}`;
      renderLinks(entry.row.querySelector(".history-links"), links);
    });
  }

  addAudioChunk(pcm) {
    if (this.current && !this.current.finished) this.current.chunks.push(pcm);
  }

  finish({ status = "", text = "", wav = null } = {}) {
    const entry = this.current;
    if (!entry || entry.finished) return;
    entry.finished = true;
    if (!wav && entry.chunks.length) wav = pcm16Wav(entry.chunks, Infinity, entry.sampleRate);
    entry.chunks = [];
    entry.text = text;
    entry.replyWav = wav ? new Blob([wav], { type: "audio/wav" }) : null;
    this.follow(() => {
      entry.status.textContent = status;
      entry.status.hidden = !status;
      const reply = entry.row.querySelector(".history-reply");
      reply.textContent = text;
      reply.hidden = !text;
      if (wav) {
        const label = `第 ${entry.id} 轮客服语音${status ? "（已收到的片段）" : ""}`;
        entry.output.setAttribute("aria-label", label);
        setPreview(entry.output, entry.replyWav);
        entry.row.querySelector(".history-output-meta").hidden = !status;
      }
      entry.row.querySelector(".history-assistant").hidden = !text && !wav;
      entry.row.querySelector(".history-no-results").hidden = !entry.row.querySelector(".history-retrieval").hidden;
      entry.row.querySelector(".history-actions").hidden = !(entry.questionWav || wav || text || entry.retrieval);
    });
  }
}
