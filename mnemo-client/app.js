const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const chatWindow = document.getElementById("chatWindow");
const sendButton = document.getElementById("sendButton");
const apiBaseInput = document.getElementById("apiBase");
const voiceModeInput = document.getElementById("voiceMode");
const voiceNameInput = document.getElementById("voiceName");
const langCodeInput = document.getElementById("langCode");
const speedInput = document.getElementById("speed");

let audioQueue = [];
let isAudioPlaying = false;
let playbackWaiters = [];
let lastPlaybackError = null;

function addMessage(role, text) {
  const el = document.createElement("article");
  el.className = `message ${role}`;
  el.textContent = text;
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  messageInput.disabled = isBusy;
  voiceModeInput.disabled = isBusy;
  voiceNameInput.disabled = isBusy;
  langCodeInput.disabled = isBusy;
  speedInput.disabled = isBusy;
}

function notifyPlaybackIdle() {
  for (const resolve of playbackWaiters) {
    resolve();
  }
  playbackWaiters = [];
}

function waitForPlaybackIdle() {
  if (!isAudioPlaying && audioQueue.length === 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => playbackWaiters.push(resolve));
}

function playNextAudioChunk() {
  if (isAudioPlaying) {
    return;
  }

  const nextItem = audioQueue.shift();
  if (!nextItem) {
    notifyPlaybackIdle();
    return;
  }

  const nextUrl = nextItem.url;
  let started = false;

  isAudioPlaying = true;
  const audio = new Audio(nextUrl);

  const finalize = () => {
    URL.revokeObjectURL(nextUrl);
    isAudioPlaying = false;
    playNextAudioChunk();
  };

  audio.onplaying = () => {
    if (started) {
      return;
    }
    started = true;
    if (typeof nextItem.onStart === "function") {
      nextItem.onStart();
    }
  };

  audio.onended = () => finalize();
  audio.onerror = () => {
    lastPlaybackError = "Audio playback failed for one chunk.";
    finalize();
  };

  const playPromise = audio.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {
      lastPlaybackError = "Browser blocked autoplay. Click the page and retry.";
      finalize();
    });
  }
}

function queueAudioChunk(bytes, onStart) {
  const blob = new Blob([bytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  audioQueue.push({ url, onStart });
  playNextAudioChunk();
}

function decodeBase64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function parseSseBlock(block) {
  const lines = block.split(/\r?\n/);
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (!line) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  const dataText = dataLines.join("\n");
  if (!dataText) {
    return { event, data: {} };
  }

  try {
    return { event, data: JSON.parse(dataText) };
  } catch {
    return { event, data: { text: dataText } };
  }
}

function parseStreamChunk(chunk) {
  // Supports plain text, NDJSON lines, and SSE-style `data:` lines.
  const lines = chunk.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return chunk;
  }

  let output = "";
  for (const line of lines) {
    const cleaned = line.startsWith("data:") ? line.slice(5).trim() : line;
    try {
      const obj = JSON.parse(cleaned);
      if (typeof obj.response === "string") {
        output += obj.response;
      } else if (typeof obj.text === "string") {
        output += obj.text;
      } else {
        output += cleaned;
      }
    } catch {
      output += cleaned;
    }
  }
  return output;
}

async function streamChatResponse(baseUrl, userText, assistantEl) {
  const response = await fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: userText }),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      if (err?.detail) {
        detail = `${detail}: ${err.detail}`;
      }
    } catch {
      // Keep default detail when error body is not JSON.
    }
    throw new Error(detail);
  }

  if (!response.body) {
    const data = await response.json();
    assistantEl.textContent = data?.response || "";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    const splitLines = buffer.split(/\r?\n/);
    buffer = splitLines.pop() || "";
    const complete = splitLines.join("\n");
    if (complete) {
      assistantEl.textContent += parseStreamChunk(complete);
    }

    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  if (buffer) {
    assistantEl.textContent += parseStreamChunk(buffer);
  }

  if (!assistantEl.textContent.trim()) {
    assistantEl.textContent = "";
  }
}

async function streamVoiceResponse(baseUrl, payload, assistantEl) {
  const response = await fetch(`${baseUrl}/voice/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    const errorText = await response.text();
    try {
      const err = JSON.parse(errorText);
      if (err?.detail) {
        detail = `${detail}: ${err.detail}`;
      }
    } catch {
      if (errorText) {
        detail = `${detail}: ${errorText}`;
      }
    }
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("Voice stream body was not available.");
  }

  lastPlaybackError = null;
  let chunkCount = 0;
  const clauseAudioBuffers = new Map();
  const clauseTextMap = new Map();
  assistantEl.textContent = "";

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const parsed = parseSseBlock(block.trim());
      if (!parsed.event) {
        continue;
      }

      if (parsed.event === "audio") {
        if (typeof parsed.data.chunk_b64 === "string" && Number.isInteger(parsed.data.index)) {
          const clauseIndex = parsed.data.index;
          const audioBytes = decodeBase64ToBytes(parsed.data.chunk_b64);
          const list = clauseAudioBuffers.get(clauseIndex) || [];
          list.push(audioBytes);
          clauseAudioBuffers.set(clauseIndex, list);
        }
      } else if (parsed.event === "audio_clause_done") {
        if (Number.isInteger(parsed.data.index)) {
          const clauseIndex = parsed.data.index;
          if (typeof parsed.data.text === "string") {
            clauseTextMap.set(clauseIndex, parsed.data.text);
          }
          const list = clauseAudioBuffers.get(clauseIndex) || [];
          if (list.length > 0) {
            const merged = new Blob(list, { type: "audio/wav" });
            const mergedBytes = new Uint8Array(await merged.arrayBuffer());
            const clauseText = clauseTextMap.get(clauseIndex) || "";
            queueAudioChunk(mergedBytes, () => {
              assistantEl.textContent += clauseText;
              chatWindow.scrollTop = chatWindow.scrollHeight;
            });
            chunkCount += 1;
            clauseAudioBuffers.delete(clauseIndex);
            clauseTextMap.delete(clauseIndex);
          }
        }
      } else if (parsed.event === "error") {
        const message = typeof parsed.data.message === "string" ? parsed.data.message : "Voice stream error.";
        throw new Error(message);
      }
    }

    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  if (buffer.trim()) {
    const parsed = parseSseBlock(buffer.trim());
    if (parsed.event === "audio" && typeof parsed.data.chunk_b64 === "string" && Number.isInteger(parsed.data.index)) {
      const clauseIndex = parsed.data.index;
      const audioBytes = decodeBase64ToBytes(parsed.data.chunk_b64);
      const list = clauseAudioBuffers.get(clauseIndex) || [];
      list.push(audioBytes);
      clauseAudioBuffers.set(clauseIndex, list);
    }
    if (parsed.event === "audio_clause_done" && Number.isInteger(parsed.data.index)) {
      const clauseIndex = parsed.data.index;
      if (typeof parsed.data.text === "string") {
        clauseTextMap.set(clauseIndex, parsed.data.text);
      }
      const list = clauseAudioBuffers.get(clauseIndex) || [];
      if (list.length > 0) {
        const merged = new Blob(list, { type: "audio/wav" });
        const mergedBytes = new Uint8Array(await merged.arrayBuffer());
        const clauseText = clauseTextMap.get(clauseIndex) || "";
        queueAudioChunk(mergedBytes, () => {
          assistantEl.textContent += clauseText;
          chatWindow.scrollTop = chatWindow.scrollHeight;
        });
        chunkCount += 1;
        clauseAudioBuffers.delete(clauseIndex);
        clauseTextMap.delete(clauseIndex);
      }
    }
    if (parsed.event === "error") {
      const message = typeof parsed.data.message === "string" ? parsed.data.message : "Voice stream error.";
      throw new Error(message);
    }
  }

  // Flush any remaining clause buffers if the stream ended unexpectedly.
  const pendingClauseIds = Array.from(clauseAudioBuffers.keys()).sort((a, b) => a - b);
  for (const clauseIndex of pendingClauseIds) {
    const list = clauseAudioBuffers.get(clauseIndex) || [];
    if (list.length === 0) {
      continue;
    }
    const merged = new Blob(list, { type: "audio/wav" });
    const mergedBytes = new Uint8Array(await merged.arrayBuffer());
    const clauseText = clauseTextMap.get(clauseIndex) || "";
    queueAudioChunk(mergedBytes, () => {
      assistantEl.textContent += clauseText;
      chatWindow.scrollTop = chatWindow.scrollHeight;
    });
    chunkCount += 1;
    clauseTextMap.delete(clauseIndex);
  }

  await waitForPlaybackIdle();

  if (lastPlaybackError) {
    assistantEl.textContent += `\n\n[Audio warning] ${lastPlaybackError}`;
    return;
  }

  if (!assistantEl.textContent.trim()) {
    assistantEl.textContent = `(Voice stream finished with ${chunkCount} audio chunks, but no text tokens were displayed.)`;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const userText = messageInput.value.trim();
  const baseUrl = apiBaseInput.value.trim().replace(/\/$/, "");
  const useVoiceMode = voiceModeInput.checked;
  const voice = voiceNameInput.value.trim() || "af_heart";
  const langCode = (langCodeInput.value.trim() || "a").slice(0, 1);
  const speed = Number(speedInput.value);

  if (!userText || !baseUrl) {
    return;
  }

  addMessage("user", userText);
  const assistantEl = addMessage("assistant", "");
  messageInput.value = "";
  setBusy(true);

  try {
    if (useVoiceMode) {
      await streamVoiceResponse(
        baseUrl,
        {
          text: userText,
          voice,
          lang_code: langCode,
          speed: Number.isFinite(speed) ? speed : 1.0,
        },
        assistantEl,
      );
    } else {
      await streamChatResponse(baseUrl, userText, assistantEl);
    }

    if (!assistantEl.textContent.trim()) {
      assistantEl.textContent = "(No response text received)";
    }
  } catch (error) {
    assistantEl.textContent = `Error: ${error.message}`;
  } finally {
    setBusy(false);
    messageInput.focus();
  }
});
