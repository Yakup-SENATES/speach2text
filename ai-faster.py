from flask import Flask, request, jsonify, render_template
import whisper
from gtts import gTTS
from TTS.api import TTS
import torch
import logging
import os
import uuid
import subprocess
from flask_cors import CORS
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from tempfile import SpooledTemporaryFile
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

# Sabit değişkenler
UPLOAD_FOLDER = os.path.join(app.static_folder, 'audio')
MAX_MEMORY_FILE_SIZE = 10 * 1024 * 1024  # 10MB
THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# Klasör kontrolü
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model yükleme optimizasyonu
@lru_cache(maxsize=1)
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("medium").to(device)

@lru_cache(maxsize=1)
def load_tts_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as e:
        logging.error(f"TTS model loading error: {e}")
        return None

# Model instances
whisper_model = load_whisper_model()
tts_model = load_tts_model()

async def process_audio_file(audio_file):
    # SpooledTemporaryFile kullanarak bellekte işlem
    with SpooledTemporaryFile(max_size=MAX_MEMORY_FILE_SIZE, mode='wb+') as temp_file:
        await audio_file.save(temp_file)
        temp_file.seek(0)
        return await asyncio.get_event_loop().run_in_executor(
            THREAD_POOL,
            whisper_model.transcribe,
            temp_file.name
        )

async def get_deepseek_response(prompt: str) -> str | None:
    try:
        logging.info(f"[DeepSeek] Request: {prompt}")
        process = await asyncio.create_subprocess_exec(
            "ollama", "run", "mistral", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if stdout:
            response = stdout.decode().strip()
            logging.info(f"[DeepSeek] Response: {response}")
            return response
        
        logging.warning("[DeepSeek] Empty response")
        return None
    except Exception as e:
        logging.exception("[DeepSeek] Error:")
        return None

async def generate_audio_response(text: str, original_audio_path: str = None):
    try:
        temp_file = SpooledTemporaryFile(max_size=MAX_MEMORY_FILE_SIZE, mode='wb+')
        
        if tts_model:
            await asyncio.get_event_loop().run_in_executor(
                THREAD_POOL,
                tts_model.tts_to_file,
                text,
                original_audio_path,
                temp_file.name
            )
        else:
            tts = gTTS(text=text)
            await asyncio.get_event_loop().run_in_executor(
                THREAD_POOL,
                tts.save,
                temp_file.name
            )
        
        return temp_file
    except Exception as e:
        logging.exception("Audio generation error:")
        raise

@app.route('/')
async def home():
    return await asyncio.get_event_loop().run_in_executor(
        THREAD_POOL,
        render_template,
        'index2.html'
    )

@app.route('/ai_speak', methods=['POST'])
async def ai_speak():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    try:
        audio_file = request.files['audio']
        
        # Transcribe audio
        result = await process_audio_file(audio_file)
        user_input = result["text"]

        # Get AI response
        deepseek_response = await get_deepseek_response(user_input)
        if not deepseek_response:
            return jsonify({'error': 'DeepSeek failed to respond'}), 500

        # Generate audio response
        filename_tts = f"{uuid.uuid4()}.mp3"
        filepath_tts = os.path.join(UPLOAD_FOLDER, filename_tts)
        
        audio_temp_file = await generate_audio_response(deepseek_response)
        
        # Save to disk efficiently
        async with aiofiles.open(filepath_tts, 'wb') as f:
            audio_temp_file.seek(0)
            await f.write(audio_temp_file.read())
        
        audio_temp_file.close()
        
        return jsonify({'audio_url': f'/static/audio/{filename_tts}'})

    except Exception as e:
        logging.exception("Error in ai_speak:")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)