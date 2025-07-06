from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from shared_config import HF_TOKEN, LLaMA_MODEL_NAME, PODCAST_AUDIO_DIR
import edge_tts, os, asyncio
from datetime import datetime
from pydub import AudioSegment

# ====== ✅ Load the model and tokenizer ONCE globally ======
tokenizer = AutoTokenizer.from_pretrained(LLaMA_MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    LLaMA_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    token=HF_TOKEN
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# ===========================================================

def summarize(merge_docs: str) -> str:
    prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>GUIDANCE:<|end_header_id|>
    You are given some recent news content.

    Write a fluent and complete summary in paragraph form, ensuring that all key details and important facts are preserved.
    Avoid using bullet points or lists.
    Your response should sound like a news anchor or reporter explaining the news to a general audience.

    Here is the news content:
    {merge_docs}
    <|start_header_id|>MODEL RESPONSE:<|end_header_id|>
    """
    return pipe(prompt, max_new_tokens=700, return_full_text=False, do_sample=False)[0]["generated_text"]

def generate_script(summary: str) -> str:
    prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>GUIDANCE:<|end_header_id|>
    You are writing a podcast script featuring only two co-hosts: **Ayesha** and **Sam**. Do not include any other characters or speakers.

    - They take turns presenting ideas.
    - Sometimes one finishes or adds onto the other’s sentence.
    - They react to each other’s thoughts with curiosity, agreement, or additional facts.
    - The tone should be friendly, conversational, and informative like two well-informed people exploring a topic together.
    - Ensure that all key points from the summary are included in their discussion.
    - Wrap up the conversation naturally at the end.

    Summary:
    {summary}

    <|start_header_id|>MODEL RESPONSE:<|end_header_id|>
    """
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
