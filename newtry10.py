import ctypes
import os
import re
import json
import time
import queue
import threading
import tempfile
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import numpy as np
import pvporcupine
import pyaudio
import requests
import sounddevice as sd
import websockets
import asyncio
import webrtcvad
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io.wavfile import write
from kokoro_onnx import Kokoro
import soundfile as sf
from scipy.signal import resample, resample_poly

# === DeepFilterNet 3 Noise Suppression ===
from df.enhance import enhance, init_df
df_state = init_df()

# === pyaec Echo Cancellation DLL (Windows) ===
dll_path = os.path.abspath("aec.dll")
ctypes.CDLL(dll_path)
print("[AEC] Echo Cancellation DLL loaded!")

import pyaec

class EchoCanceller:
    def __init__(self, sample_rate=48000, frame_size=480, filter_length=128):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.state = pyaec.Aec(frame_size=frame_size, filter_length=filter_length, sample_rate=sample_rate)

    def cancel_echo(self, mic, ref):
        return self.state.cancel_echo(mic, ref)

MIC_DEVICE_INDEX = 60  # Set to your mic index!
MIC_SAMPLE_RATE = 48000
FRAME_SIZE = 480
aec = EchoCanceller(sample_rate=MIC_SAMPLE_RATE, frame_size=FRAME_SIZE)
ref_audio_buffer = np.zeros(FRAME_SIZE, dtype=np.int16)
ref_audio_lock = threading.Lock()

def set_ref_audio(audio_chunk):
    global ref_audio_buffer
    arr = np.frombuffer(audio_chunk, dtype=np.int16)
    with ref_audio_lock:
        if arr.size >= FRAME_SIZE:
            ref_audio_buffer = arr[-FRAME_SIZE:]
        else:
            ref_audio_buffer = np.pad(arr, (FRAME_SIZE-arr.size, 0), 'constant')

def cancel_mic_audio(mic_chunk):
    mic = np.frombuffer(mic_chunk, dtype=np.int16)[:FRAME_SIZE]
    with ref_audio_lock:
        ref_copy = ref_audio_buffer.copy()
    result = aec.cancel_echo(mic, ref_copy)
    if isinstance(result, list):
        result = np.array(result)
    return result.astype(np.int16).tobytes()

def downsample(audio, orig_sr, target_sr):
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    audio_resampled = resample_poly(audio, up, down)
    audio_resampled = np.clip(audio_resampled, -1.0, 1.0)
    return (audio_resampled * 32767).astype(np.int16)

# === Whisper WebSocket Streaming Integration ===
FASTER_WHISPER_WS = "ws://localhost:9090"

def stt_stream(audio):
    async def ws_stt(audio):
        try:
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            print(f"[DEBUG] Sending audio with shape {audio.shape}, dtype: {audio.dtype}, max: {audio.max()}, min: {audio.min()}")
            async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
                await ws.send(audio.tobytes())
                await ws.send("end")
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=18)
                except asyncio.TimeoutError:
                    print("[Buddy] Whisper timeout. Brak odpowiedzi przez 18s.")
                    return ""
                text = message.decode("utf-8") if isinstance(message, bytes) else message
                print(f"\n[Buddy] === Whisper rozpoznał: \"{text}\" ===")
                return text
        except Exception as e:
            print(f"[Buddy] Błąd połączenia z Whisper: {e}")
            return ""
    return asyncio.run(ws_stt(audio))

tts_queue = queue.Queue()
playback_queue = queue.Queue()
current_playback = None
playback_stop_flag = threading.Event()
buddy_talking = threading.Event()
vad_triggered = threading.Event()
LAST_FEW_BUDDY = []
RECENT_WHISPER = []

DEFAULT_LANG = "en"
FAST_MODE = True
DEBUG = True
device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
if DEBUG:
    print(f"[Buddy] Running on device: {device}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
print("Embedding model loaded", flush=True)

CHIME_PATH = "chime.wav"
known_users_path = "known_users.json"
THEMES_PATH = "themes_memory"
LAST_USER_PATH = "last_user.json"
os.makedirs(THEMES_PATH, exist_ok=True)

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
WEATHERAPI_ENDPOINT = "http://api.weatherapi.com/v1/current.json"
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")

KOKORO_VOICES = {
    "pl": "af_heart",
    "en": "af_heart",
    "it": "if_sara",
}
KOKORO_LANGS = {
    "pl": "pl",
    "en": "en-us",
    "it": "it"
}
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print("Kokoro loaded", flush=True)

if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
else:
    known_users = {}

# ========= USER MEMORY (NEW STRUCTURED MEMORY) =========
def get_user_memory_path(name):
    return f"user_memory_{name}.json"

def load_user_memory(name):
    path = get_user_memory_path(name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_memory(name, memory):
    path = get_user_memory_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def update_user_memory(name, utterance):
    memory = load_user_memory(name)
    text = utterance.lower()

    # Mood
    if re.search(r"\bi('?m| am| feel) sad\b", text):
        memory["mood"] = "sad"
    elif re.search(r"\bi('?m| am| feel) happy\b", text):
        memory["mood"] = "happy"
    elif re.search(r"\bi('?m| am| feel) (angry|mad|upset)\b", text):
        memory["mood"] = "angry"

    # Hobbies/interests (Marvel movies)
    if re.search(r"\bi (love|like|enjoy|prefer) (marvel movies|marvel|comics)\b", text):
        hobbies = memory.get("hobbies", [])
        if "marvel movies" not in hobbies:
            hobbies.append("marvel movies")
        memory["hobbies"] = hobbies

    # Work issues
    if "issue at work" in text or "problems at work" in text or "problem at work" in text:
        memory["work_issue"] = "open"
    if ("issue" in memory and "solved" in text) or ("work_issue" in memory and ("solved" in text or "fixed" in text)):
        memory["work_issue"] = "resolved"

    # Add more phrase matching as needed for more details

    save_user_memory(name, memory)

def build_user_facts(name):
    memory = load_user_memory(name)
    facts = []
    if "mood" in memory:
        facts.append(f"The user was previously {memory['mood']}.")
    if "hobbies" in memory:
        facts.append(f"The user likes: {', '.join(memory['hobbies'])}.")
    if memory.get("work_issue") == "open":
        facts.append(f"The user had unresolved issues at work.")
    return facts

# ========= END USER MEMORY ==========

def tts_worker():
    while True:
        item = tts_queue.get()
        if item is None:
            break
        if isinstance(item, tuple):
            if len(item) == 2:
                text, lang = item
                style = {}
            else:
                text, lang, style = item
        else:
            text, lang, style = item, "en", {}
        try:
            if text.strip():
                generate_and_play_kokoro(text, lang)
        except Exception as e:
            print(f"[TTS Error] {e}")
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def stop_playback():
    global current_playback
    playback_stop_flag.set()
    if current_playback and hasattr(current_playback, "is_playing") and current_playback.is_playing():
        current_playback.stop()
        current_playback = None
    while not playback_queue.empty():
        try:
            playback_queue.get_nowait()
            playback_queue.task_done()
        except queue.Empty:
            break

def listen_for_stopword():
    vad = webrtcvad.Vad(2)
    blocksize = int(MIC_SAMPLE_RATE * 0.02)
    silence_timeout = 0.8
    frames = []
    start_time = time.time()
    try:
        with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, channels=1, dtype='int16', blocksize=blocksize) as stream:
            while buddy_talking.is_set():
                frame, _ = stream.read(blocksize)
                frames.append(frame)
                if len(frames) > int(MIC_SAMPLE_RATE / blocksize * 1.2):
                    frames = frames[-int(MIC_SAMPLE_RATE / blocksize * 1.2):]
                if not vad.is_speech(downsample(frame.flatten(), MIC_SAMPLE_RATE, 16000).tobytes(), 16000):
                    if time.time() - start_time > silence_timeout:
                        frames = []
                        start_time = time.time()
                        continue
                else:
                    start_time = time.time()
                if len(frames) * blocksize >= MIC_SAMPLE_RATE:
                    audio_np = np.concatenate(frames, axis=0).astype(np.int16)
                    audio_np_16k = downsample(audio_np, MIC_SAMPLE_RATE, 16000)
                    text = stt_stream(audio_np_16k)
                    if text and "buddy stop" in text.lower():
                        if DEBUG:
                            print("[Buddy][STOPWORD] Detected 'buddy stop'! Interrupting.")
                        stop_playback()
                        break
    except Exception as e:
        if DEBUG:
            print(f"[Buddy][STOPWORD] Error: {e}")

def audio_playback_worker():
    global current_playback
    while True:
        audio = playback_queue.get()
        if audio is None:
            break
        try:
            # --- Echo cancellation: set reference audio for AEC every time we play audio
            set_ref_audio(audio.raw_data if hasattr(audio, "raw_data") else audio._data)
            playback_stop_flag.clear()
            buddy_talking.set()
            current_playback = _play_with_simpleaudio(audio)
            while current_playback and current_playback.is_playing():
                if playback_stop_flag.is_set():
                    current_playback.stop()
                    break
                time.sleep(0.05)
            current_playback = None
        except Exception as e:
            print(f"[Buddy] Audio playback error: {e}")
        finally:
            buddy_talking.clear()
            playback_queue.task_done()

threading.Thread(target=audio_playback_worker, daemon=True).start()

def wait_after_buddy_speaks(delay=1.2):
    playback_queue.join()
    while buddy_talking.is_set():
        time.sleep(0.05)
    time.sleep(delay)

def vad_and_listen():
    vad = webrtcvad.Vad(3)
    blocksize = int(MIC_SAMPLE_RATE * 0.02)
    blocksize_16k = 320
    with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, channels=1, blocksize=blocksize, dtype='int16') as stream:
        print("\n[Buddy] === SŁUCHAM, mów do mnie... ===")
        silence_thresh = 1.0
        min_speech_frames = 10
        frame_buffer = []
        speech_detected = 0
        while True:
            frame, _ = stream.read(blocksize)
            mic_frame_aec = cancel_mic_audio(frame.tobytes())
            mic_frame_aec = np.frombuffer(mic_frame_aec, dtype=np.int16)
            frame_16k = resample(mic_frame_aec, blocksize_16k).astype(np.int16)
            if vad.is_speech(frame_16k.tobytes(), 16000):
                frame_buffer.append(mic_frame_aec)
                speech_detected += 1
                if speech_detected >= min_speech_frames:
                    print("[Buddy] VAD: Wykryto mowę. Nagrywam...")
                    audio = frame_buffer.copy()
                    last_speech = time.time()
                    start_time = time.time()
                    frame_buffer.clear()
                    while time.time() - last_speech < silence_thresh and (time.time() - start_time) < 8:
                        frame, _ = stream.read(blocksize)
                        mic_frame_aec = cancel_mic_audio(frame.tobytes())
                        mic_frame_aec = np.frombuffer(mic_frame_aec, dtype=np.int16)
                        frame_16k = resample(mic_frame_aec, blocksize_16k).astype(np.int16)
                        audio.append(mic_frame_aec)
                        if vad.is_speech(frame_16k.tobytes(), 16000):
                            last_speech = time.time()
                    print("[Buddy] Koniec nagrania. Wysyłam do Whisper...")

                    audio_np = np.concatenate(audio, axis=0).astype(np.int16)
                    audio_np_16k = downsample(audio_np, MIC_SAMPLE_RATE, 16000)
                    try:
                        audio_np_16k = enhance(df_state, audio_np_16k)
                        print("[Buddy][DeepFilterNet] Enhancement applied.")
                    except Exception as e:
                        print(f"[Buddy][DeepFilterNet] Enhancement failed: {e}")
                    print("[DEBUG] audio_np_16k shape:", audio_np_16k.shape, "dtype:", audio_np_16k.dtype, "min:", np.min(audio_np_16k), "max:", np.max(audio_np_16k))
                    return audio_np_16k.astype(np.int16)
            else:
                if len(frame_buffer) > 0:
                    frame_buffer.clear()
                speech_detected = 0

def fast_listen_and_transcribe():
    wait_after_buddy_speaks()
    audio = vad_and_listen()
    try:
        print("[DEBUG] Saving temp_input.wav, shape:", audio.shape, "dtype:", audio.dtype, "min:", np.min(audio), "max:", np.max(audio))
        write("temp_input.wav", 16000, audio)
        info = sf.info("temp_input.wav")
        print("[DEBUG] temp_input.wav info:", info)
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Błąd przy zapisie temp_input.wav: {e}")
    text = stt_stream(audio)
    cleaned = re.sub(r'[^\w\s]', '', text.strip().lower())
    if cleaned:
        RECENT_WHISPER.append(cleaned)
        if len(RECENT_WHISPER) > 5:
            RECENT_WHISPER.pop(0)
    if is_noise_or_gibberish(text):
        return ""
    return text

def is_noise_or_gibberish(text):
    cleaned = text.strip().lower()
    return not cleaned

def speak_async(text, lang=DEFAULT_LANG, style=None):
    if not text.strip():
        return
    tts_queue.put((text.strip(), lang, style or {}))

def play_chime():
    try:
        audio = AudioSegment.from_wav(CHIME_PATH)
        playback_queue.put(audio)
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Error playing chime: {e}")

def save_known_users():
    with open(known_users_path, "w", encoding="utf-8") as f:
        json.dump(known_users, f, indent=2, ensure_ascii=False)

def load_user_history(name):
    path = f"history_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_history(name, history):
    path = f"history_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, ensure_ascii=False, indent=2)

def extract_topic_from_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for w in words:
        if len(w) < 4:
            continue
        freq[w] = freq.get(w, 0) + 1
    if freq:
        return max(freq, key=freq.get)
    return None

def update_thematic_memory(user, utterance):
    topic = extract_topic_from_text(utterance)
    if not topic:
        return
    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    if os.path.exists(theme_path):
        with open(theme_path, "r", encoding="utf-8") as f:
            themes = json.load(f)
    else:
        themes = {}
    themes[topic] = themes.get(topic, 0) + 1
    with open(theme_path, "w", encoding="utf-8") as f:
        json.dump(themes, f, ensure_ascii=False, indent=2)

def get_frequent_topics(user, top_n=3):
    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    if not os.path.exists(theme_path):
        return []
    with open(theme_path, "r", encoding="utf-8") as f:
        themes = json.load(f)
    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_themes[:top_n]]

def generate_embedding(text):
    return embedding_model.encode([text])[0]

def match_known_user(new_embedding, threshold=0.75):
    best_name, best_score = None, 0
    for name, emb in known_users.items():
        sim = cosine_similarity([new_embedding], [emb])[0][0]
        if sim > best_score:
            best_name, best_score = name, sim
    return (best_name, best_score) if best_score >= threshold else (None, best_score)

def get_last_user():
    if os.path.exists(LAST_USER_PATH):
        try:
            with open(LAST_USER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)["name"]
        except Exception:
            return None
    return None

def set_last_user(name):
    with open(LAST_USER_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name}, f)

def identify_or_register_user():
    if FAST_MODE:
        return "Guest"
    last_user = get_last_user()
    if last_user and last_user in known_users:
        if DEBUG:
            print(f"[Buddy] Welcome back, {last_user}!")
        return last_user
    speak_async("Cześć! Jak masz na imię?", "pl")
    speak_async("Hi! What's your name?", "en")
    speak_async("Ciao! Come ti chiami?", "it")
    playback_queue.join()
    name = fast_listen_and_transcribe().strip().title()
    if not name:
        name = f"User{int(time.time())}"
    known_users[name] = generate_embedding(name).tolist()
    save_known_users()
    set_last_user(name)
    speak_async(f"Miło Cię poznać, {name}!", lang="pl")
    playback_queue.join()
    return name

def build_personality_prompt(tone):
    personality_map = {
        "friendly": "You're a warm and witty assistant named Buddy. You respond with a friendly, conversational tone, often adding light humor or personal touches.",
        "professional": "You're a concise and professional assistant named Buddy. You provide clear, accurate answers with a respectful tone.",
        "neutral": "You're a helpful assistant named Buddy. You respond naturally and clearly in a neutral tone."
    }
    personality_desc = personality_map.get(tone, personality_map["neutral"])
    return f"""{personality_desc}
Always answer like you're talking to a real person. Avoid robotic phrasing.
Keep responses engaging. If the user sounds confused or emotional, show empathy.
Your responses can use short interjections like 'Hmm', 'I see', 'Got it!', 'Great question!'.
If possible, keep your answers concise (1-2 sentences), unless the user asks for more detail.
"""

def decide_reply_length(question, conversation_mode="auto"):
    short_triggers = ["what time", "who", "quick", "fast", "short", "how many", "when"]
    long_triggers = ["explain", "describe", "details", "why", "history", "story"]
    q = question.lower()
    if conversation_mode == "fast":
        return "short"
    if conversation_mode == "long":
        return "long"
    if any(t in q for t in short_triggers):
        return "short"
    if any(t in q for t in long_triggers):
        return "long"
    return "long" if len(q.split()) > 8 else "short"

def should_get_weather(question):
    q = question.lower().strip()
    weather_keywords = ["weather", "pogoda", "temperature", "temperatura"]
    question_starters = ["what", "jaka", "jaki", "jakie", "czy", "is", "how", "when", "where", "will"]
    is_question = (
        "?" in q
        or any(q.startswith(w + " ") for w in question_starters)
        or q.endswith(("?",))
    )
    return is_question and any(k in q for k in weather_keywords)

def build_openai_messages(name, tone_style, history, question, lang, topics, reply_length):
    personality = build_personality_prompt(tone_style)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    sys_msg = f"""{personality}
IMPORTANT: Always answer in {lang_name}. Never switch language unless user does.
Always respond in plain text—never use markdown, code blocks, or formatting.
"""
    # Inject structured user facts!
    facts = build_user_facts(name)
    if topics:
        sys_msg += f"You remember these user interests/topics: {', '.join(topics)}.\n"
    if facts:
        sys_msg += "Known facts about the user: " + " ".join(facts) + "\n"
    messages = [
        {"role": "system", "content": sys_msg}
    ]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["buddy"]})
    messages.append({"role": "user", "content": question})
    return messages

def extract_last_buddy_reply(full_text):
    matches = list(re.finditer(r"Buddy:", full_text, re.IGNORECASE))
    if matches:
        last = matches[-1].end()
        reply = full_text[last:].strip()
        reply = re.split(r"(?:User:|Buddy:)", reply)[0].strip()
        reply = re.sub(r"^`{3,}.*?`{3,}$", "", reply, flags=re.DOTALL|re.MULTILINE)
        return reply if reply else full_text.strip()
    return full_text.strip()

def should_end_conversation(text):
    end_phrases = [
        "koniec", "do widzenia", "dziękuję", "thanks", "bye", "goodbye", "that's all", "quit", "exit"
    ]
    if not text:
        return False
    lower = text.strip().lower()
    return any(phrase in lower for phrase in end_phrases)

def stream_chunks_smart(text, max_words=20):
    buffer = text.strip()
    chunks = []
    sentences = re.findall(r'.+?[.!?](?=\s|$)', buffer)
    remainder = re.sub(r'.+?[.!?](?=\s|$)', '', buffer).strip()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk.strip())
        else:
            chunks.append(sentence)
    return chunks, remainder

def ask_llama3_openai_streaming(messages, model="llama3", max_tokens=60, temperature=0.5, lang="en", style=None):
    url = "http://localhost:5001/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            buffer = ""
            full_response = ""
            already_spoken = set()

            def speak_new_chunks(text):
                chunks, leftover = stream_chunks_smart(text)
                for chunk in chunks:
                    normalized = chunk.lower().strip()
                    if normalized not in already_spoken and len(chunk.split()) >= 2:
                        print(f"\n[Buddy] ==>> Buddy mówi: {chunk}")
                        speak_async(chunk, lang=lang, style=style)
                        time.sleep(0.35)
                        already_spoken.add(normalized)
                return leftover

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    s = line.decode("utf-8").strip()
                    if not s:
                        continue
                    if s.startswith("data:"):
                        s = s[5:].strip()
                    if s == "[DONE]":
                        break
                    data = json.loads(s)
                    delta = ""
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {}).get("content", "") \
                            or data["choices"][0].get("message", {}).get("content", "")
                    elif "content" in data:
                        delta = data["content"]
                    if delta:
                        print(delta, end="", flush=True)
                        buffer += delta
                        full_response += delta
                        if any(p in delta for p in ".!?") or len(buffer) > 50:
                            buffer = speak_new_chunks(buffer)
                except Exception as err:
                    print(f"\n[Buddy][Stream JSON Error] {err}")

            if buffer.strip():
                leftover = speak_new_chunks(buffer)
                if leftover.strip():
                    print(f"[Buddy] ==>> Buddy mówi: {leftover}")
                    speak_async(leftover, lang=lang, style=style)

            return full_response.strip()

    except Exception as e:
        print(f"[Buddy][OpenAI Streaming Error] {e}")
        return ""

def handle_user_interaction(speaker, history, conversation_mode="auto"):
    global LAST_FEW_BUDDY
    wait_after_buddy_speaks(delay=0.2)
    if DEBUG:
        print("[Buddy] Active conversation. Speak when ready!")
    vad_triggered.clear()
    while buddy_talking.is_set():
        time.sleep(0.05)
    question = fast_listen_and_transcribe()
    print(f"[DEBUG] Rozpoznano pytanie: {question!r}")
    play_chime()
    lang = detect_language(question)
    if DEBUG:
        print(f"[Buddy] Detected language: {lang}")
    if vad_triggered.is_set():
        if DEBUG:
            print("[Buddy] Barage-in: live TTS stopped, moving to new question.")
        vad_triggered.clear()
    if not question:
        print("[DEBUG] PUSTE PYTANIE, wychodzę z obsługi interakcji.")
        return True
    if is_noise_or_gibberish(question):
        print(f"[DEBUG] ODRZUCONO JAKO GIBBERISH: {question!r}")
        return True
    INTERRUPT_PHRASES = ["stop", "buddy stop", "przerwij", "cancel"]
    if any(phrase in question.lower() for phrase in INTERRUPT_PHRASES):
        if DEBUG:
            print("[Buddy] Received interrupt command.")
        stop_playback()
        return True
    if should_end_conversation(question):
        if DEBUG:
            print("[Buddy] Ending conversation as requested.")
        return False
    style = {"emotion": "neutral"}
    if should_get_weather(question):
        location = extract_location_from_question(question)
        forecast = get_weather(location, lang)
        print(f"[DEBUG] Prognoza pogody wygenerowana: {forecast!r}")
        speak_async(forecast, lang, style)
        return True
    if should_handle_homeassistant(question):
        answer = handle_homeassistant_command(question)
        if answer:
            print(f"[DEBUG] Home Assistant odpowiedź: {answer!r}")
            speak_async(answer, lang, style)
            return True
    if should_search_internet(question):
        result = search_internet(question, lang)
        print(f"[DEBUG] Wynik wyszukiwania internetowego: {result!r}")
        speak_async(result, lang, style)
        return True

    # ========== UPDATE USER MEMORY ON EVERY UTTERANCE ==========
    update_user_memory(speaker, question)
    # ========== ==========

    print(f"[DEBUG] Przekazuję do LLM: {question!r}")
    llm_start_time = time.time()
    ask_llama3_streaming(question, speaker, history, lang, conversation_mode, style=style)
    if DEBUG:
        print(f"[TIMING] LLM generation time: {time.time() - llm_start_time:.2f} seconds")
    return True

def ask_llama3_streaming(question, name, history, lang=DEFAULT_LANG, conversation_mode="auto", style=None):
    update_thematic_memory(name, question)
    topics = get_frequent_topics(name, top_n=3)
    user_tones = {
        "Dawid": "friendly",
        "Anna": "professional",
        "Guest": "neutral"
    }
    tone_style = user_tones.get(name, "neutral")
    reply_length = decide_reply_length(question, conversation_mode)
    messages = build_openai_messages(
        name, tone_style, history, question, lang, topics, reply_length
    )
    full_text = ""
    try:
        if DEBUG:
            print("[Buddy][LLM] FINAL MESSAGES DELIMITED BELOW:\n" + "="*40)
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print("="*40)
        full_text = ask_llama3_openai_streaming(messages, model="llama3", max_tokens=60, temperature=0.5, lang=lang, style=style)
        if DEBUG:
            print("[Buddy][LLM] RAW OUTPUT DELIMITED BELOW:\n" + "="*40)
            print(full_text)
            print("="*40)
    except Exception as e:
        if DEBUG:
            print("[Buddy] LLM HTTP error:", e)
    tts_start_time = time.time()
    if full_text.strip():
        buddy_only = extract_last_buddy_reply(full_text)
        if DEBUG:
            print("[Buddy][LLM] EXTRACTED BUDDY REPLY:", buddy_only)
        if not buddy_only.strip():
            print("[ERROR] LLM returned empty or unparseable output!")
    else:
        print("[Buddy][TTS] Skipping TTS because LLM output is empty.")
    if DEBUG:
        print(f"[TIMING] Passed to TTS in: {time.time() - tts_start_time:.2f} seconds")
    history.append({"user": question, "buddy": full_text})
    if not FAST_MODE:
        save_user_history(name, history)

def should_search_internet(question):
    triggers = [
        "szukaj w internecie", "sprawdź w internecie", "co to jest", "dlaczego", "jak zrobić",
        "what is", "why", "how to", "search the internet", "find online"
    ]
    q = question.lower()
    return any(t in q for t in triggers)

def search_internet(question, lang):
    params = {
        "q": question,
        "api_key": SERPAPI_KEY,
        "hl": lang
    }
    try:
        r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        if "answer_box" in data and "answer" in data["answer_box"]:
            return data["answer_box"]["answer"]
        if "organic_results" in data and len(data["organic_results"]) > 0:
            return data["organic_results"][0].get("snippet", "No answer found.")
        return "No answer found."
    except Exception as e:
        if DEBUG:
            print("[Buddy] SerpAPI error:", e)
        return "Unable to check the Internet now."

def get_weather(location="Warsaw", lang="en"):
    key = os.environ.get("WEATHERAPI_KEY", "YOUR_FALLBACK_KEY")
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": key,
        "q": location,
        "lang": lang
    }
    try:
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        desc = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        feels = data["current"]["feelslike_c"]
        city = data["location"]["name"]
        return f"Weather in {city}: {desc}, temperature {temp}°C, feels like {feels}°C."
    except Exception as e:
        if DEBUG:
            print("[Buddy] WeatherAPI error:", e)
        return "Unable to check the weather now."

def extract_location_from_question(question):
    match = re.search(r"(w|in|dla)\s+([A-Za-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ\s\-]+)", question, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return "Warsaw"

def should_handle_homeassistant(question):
    q = question.lower()
    keywords = ["turn on the light", "włącz światło", "zapal światło", "turn off the light", "wyłącz światło", "spotify", "youtube", "smarttube", "odtwórz"]
    return any(k in q for k in keywords)

def handle_homeassistant_command(question):
    q = question.lower()
    if "turn on the light" in q or "włącz światło" in q or "zapal światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_on")
        return f"Light in {room} has been turned on." if succ else f"Failed to turn on the light in {room}."
    if "turn off the light" in q or "wyłącz światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_off")
        return f"Light in {room} has been turned off." if succ else f"Failed to turn off the light in {room}."
    if "spotify" in q:
        succ = send_homeassistant_command("media_player.spotify", "media_player.media_play")
        return "Spotify started." if succ else "Failed to start Spotify."
    if "youtube" in q or "smarttube" in q:
        succ = send_homeassistant_command("media_player.tv_salon", "media_player.select_source", {"source": "YouTube"})
        return "YouTube launched on TV." if succ else "Failed to launch YouTube on TV."
    return None

def send_homeassistant_command(entity_id, service, data=None):
    url = f"{HOME_ASSISTANT_URL}/api/services/{service.replace('.', '/')}"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "entity_id": entity_id
    }
    if data:
        payload.update(data)
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=6)
        if r.status_code in (200, 201):
            return True
        if DEBUG:
            print("[Buddy] Home Assistant error:", r.text)
        return False
    except Exception as e:
        if DEBUG:
            print("[Buddy] Home Assistant exception:", e)
        return False

def generate_and_play_kokoro(text, lang=None):
    detected_lang = lang or detect_language(text)
    voice = KOKORO_VOICES.get(detected_lang, KOKORO_VOICES["en"])
    kokoro_lang = KOKORO_LANGS.get(detected_lang, "en-us")
    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, samples, sample_rate)
            audio = AudioSegment.from_wav(f.name)
        playback_queue.put(audio)
    except Exception as e:
        if DEBUG:
            print(f"[Buddy][Kokoro] Błąd TTS: {e}")

def detect_language(text, fallback="en"):
    try:
        if not text or len(text.strip()) < 5:
            if DEBUG:
                print(f"[Buddy DEBUG] Text too short for reliable detection, defaulting to 'en'")
            return "en"
        langs = detect_langs(text)
        if DEBUG:
            print(f"[Buddy DEBUG] detect_langs for '{text}': {langs}")
        if langs:
            best = langs[0]
            if best.prob > 0.8 and best.lang in ["en", "pl", "it"]:
                return best.lang
            if any(l.lang == "en" and l.prob > 0.5 for l in langs):
                return "en"
    except Exception as e:
        if DEBUG:
            print(f"[Buddy DEBUG] langdetect error: {e}")
    return "en"

print("Main function entered!", flush=True)

def main():
    access_key = "/PLJ88d4+jDeVO4zaLFaXNkr6XLgxuG7dh+6JcraqLhWQlk3AjMy9Q=="
    keyword_paths = [r"hey-buddy_en_windows_v3_0_0.ppn"]
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=porcupine.frame_length)
    if DEBUG:
        print("[Buddy] Waiting for wake word 'Hey Buddy'...")
    in_session, session_timeout = False, 45
    speaker = None
    history = []
    last_time = 0
    try:
        while True:
            if not in_session:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                if porcupine.process(pcm) >= 0:
                    if DEBUG:
                        print("[Buddy] Wake word detected!")
                    stop_playback()
                    speaker = identify_or_register_user()
                    history = load_user_history(speaker)
                    if DEBUG:
                        print("[Buddy] Listening for next question...")
                    in_session = handle_user_interaction(speaker, history)
                    last_time = time.time()
            else:
                if time.time() - last_time > session_timeout:
                    if DEBUG:
                        print("[Buddy] Session expired.")
                    in_session = False
                    continue
                if DEBUG:
                    print("[Buddy] Listening for next question...")
                stop_playback()
                playback_queue.join()
                while buddy_talking.is_set():
                    time.sleep(0.05)
                in_session = handle_user_interaction(speaker, history)
                last_time = time.time()
    except KeyboardInterrupt:
        if DEBUG:
            print("[Buddy] Interrupted by user.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass
        try:
            porcupine.delete()
        except Exception:
            pass
        executor.shutdown(wait=True)

if __name__ == "__main__":
    main()