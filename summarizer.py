# app/summarizer.py

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from app.shared_config import HF_TOKEN, LLaMA_MODEL_NAME, PODCAST_AUDIO_DIR
import edge_tts, os, asyncio
from datetime import datetime
from pydub import AudioSegment

def load_llama_pipe():
    tokenizer = AutoTokenizer.from_pretrained(LLaMA_MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(LLaMA_MODEL_NAME, torch_dtype="auto", device_map="auto", token=HF_TOKEN)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return pipe

def summarize(merge_docs):
    pipe = load_llama_pipe()
    prompt = f"...{merge_docs}..."
    return pipe(prompt, max_new_tokens=700, return_full_text=False, do_sample=False)[0]["generated_text"]

def generate_script(summary):
    pipe = load_llama_pipe()
    prompt = f"...{summary}..."
    return pipe(prompt, max_new_tokens=1000, return_full_text=False, do_sample=False)[0]["generated_text"]

async def synthesize_podcast(script: str):
    os.makedirs(PODCAST_AUDIO_DIR, exist_ok=True)
    lines = [("Ayesha", "en-US-AvaMultilingualNeural", t.strip()) if "Ayesha:" in t else 
             ("Sam", "en-US-AndrewMultilingualNeural", t.strip()) for t in script.split("\n") if ":" in t]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(PODCAST_AUDIO_DIR, f"podcast_{timestamp}.mp3")
    tasks = [edge_tts.Communicate(text=t, voice=v).save(f"line_{i}.mp3") for i, (_, v, t) in enumerate(lines)]
    await asyncio.gather(*tasks)

    combined = AudioSegment.silent(duration=500)
    for i in range(len(lines)):
        combined += AudioSegment.from_file(f"line_{i}.mp3") + AudioSegment.silent(duration=500)
        os.remove(f"line_{i}.mp3")
    combined.export(output_file, format="mp3")
    return os.path.basename(output_file)
