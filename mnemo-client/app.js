const chatWindow = document.getElementById("chatWindow");
const recordButton = document.getElementById("recordButton");
const apiBaseInput = document.getElementById("apiBase");
const voiceNameInput = document.getElementById("voiceName");
const langCodeInput = document.getElementById("langCode");
const speedInput = document.getElementById("speed");

let audioQueue = [];
let isAudioPlaying = false;
let playbackWaiters = [];
let lastPlaybackError = null;

// Voice recording state
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function addMessage(role, text) {
  const el = document.createElement("article");
  el.className = `message ${role}`;
  el.textContent = text;
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

function setBusy(isBusy) {
  recordButton.disabled = isBusy;
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

async function streamVoiceResponse(baseUrl, payload, assistantEl, sttMessageEl = null) {
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
        // Handle validation errors (422)
        if (Array.isArray(err.detail)) {
          detail = `${detail}: ${err.detail.map(e => `${e.loc.join('.')} - ${e.msg}`).join(', ')}`;
        } else {
          detail = `${detail}: ${err.detail}`;
        }
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

      if (parsed.event === "stt_done") {
        // Display the transcribed text from STT
        if (typeof parsed.data.text === "string" && sttMessageEl) {
          sttMessageEl.textContent = parsed.data.text;
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
      } else if (parsed.event === "audio") {
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



// Voice recording functionality
recordButton.addEventListener("click", async () => {
  if (isRecording) {
    // Stop recording
    stopRecording();
  } else {
    // Start recording
    await startRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Use webm format which is widely supported
    const options = { mimeType: 'audio/webm' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options.mimeType = 'audio/ogg';
    }
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options.mimeType = 'audio/wav';
    }
    
    mediaRecorder = new MediaRecorder(stream, options);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      await sendVoiceMessage(audioBlob);
      
      // Stop all tracks
      stream.getTracks().forEach(track => track.stop());
    };
    
    mediaRecorder.start();
    isRecording = true;
    recordButton.textContent = "⏹️ Stop Recording";
    recordButton.classList.add("recording");
  } catch (error) {
    alert(`Microphone access denied or unavailable: ${error.message}`);
    console.error("Recording error:", error);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    isRecording = false;
    recordButton.textContent = "🎤 Hold to Talk";
    recordButton.classList.remove("recording");
  }
}

async function sendVoiceMessage(audioBlob) {
  const baseUrl = apiBaseInput.value.trim().replace(/\/$/, "");
  const voice = voiceNameInput.value.trim() || "af_heart";
  const langCode = (langCodeInput.value.trim() || "a").slice(0, 1);
  const speed = Number(speedInput.value);
  
  if (!baseUrl) {
    alert("Please set the API Base URL");
    return;
  }
  
  setBusy(true);
  
  // Show user message as "🎤 Voice Message"
  const userEl = addMessage("user", "🎤 Voice Message (transcribing...)");
  const assistantEl = addMessage("assistant", "");
  
  try {
    // Convert audio blob to base64
    const arrayBuffer = await audioBlob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    let binaryString = '';
    for (let i = 0; i < bytes.length; i++) {
      binaryString += String.fromCharCode(bytes[i]);
    }
    const audioB64 = btoa(binaryString);
    
    const payload = {
      audio_b64: audioB64,
      voice,
      lang_code: langCode,
      speed: Number.isFinite(speed) ? speed : 1.0,
    };
    
    console.log('Sending voice payload:', {
      ...payload,
      audio_b64: `${audioB64.substring(0, 50)}... (${audioB64.length} chars)`
    });
    
    // Send to backend with audio_b64
    await streamVoiceResponse(
      baseUrl,
      payload,
      assistantEl,
      userEl // Pass user element to update with transcription
    );
    
    if (!assistantEl.textContent.trim()) {
      assistantEl.textContent = "(No response text received)";
    }
  } catch (error) {
    assistantEl.textContent = `Error: ${error.message}`;
    console.error("Voice message error:", error);
  } finally {
    setBusy(false);
  }
}
