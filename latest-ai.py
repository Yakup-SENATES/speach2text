from flask import Flask, request, jsonify, render_template
import whisper
from gtts import gTTS
from TTS.api import TTS
import torch
import os
import uuid
import subprocess
from flask_cors import CORS
from datetime import datetime
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium").to(device)
tts_model = None

try:
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", weights_only = False).to(device)
    logger.info("XTTS model loaded successfully")
except Exception as e:
    logger.warning(f"XTTS model failed to load, falling back to gTTS: {str(e)}")

# Chat History Storage
chat_history = []

# Helper Functions
def get_deepseek_response(prompt: str) -> str:
    """Get response from Ollama's Mistral model"""
    try:
        command = ["ollama", "run", "falcon", prompt]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30  # 30 seconds timeout
        )
        return result.stdout.strip() if result.stdout else "Yanıt alınamadı"
    except subprocess.TimeoutExpired:
        logger.error("Ollama timeout occurred")
        return "AI yanıt vermedi (timeout)"
    except Exception as e:
        logger.error(f"Ollama error: {str(e)}")
        return f"AI hatası: {str(e)}"

# Routes
@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    
    # Clear audio files (optional)
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
    
    return jsonify({'status': 'success', 'message': 'Sohbet geçmişi temizlendi'})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify({'chat_history': chat_history})

@app.route('/ai_speak', methods=['POST'])
def ai_speak():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    # Save audio file
    audio_file = request.files['audio']
    input_filename = f"input_{uuid.uuid4()}.wav"
    input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
    audio_file.save(input_filepath)

    try:
        # Step 1: Speech-to-Text
        logger.info("Transcribing audio...")
        transcription = whisper_model.transcribe(input_filepath)
        user_input = transcription.get("text", "")
        logger.info(f"Transcription: {user_input}")

        if not user_input.strip():
            return jsonify({'error': 'Konuşma algılanamadı'}), 400

        # Step 2: Get AI Response
        logger.info("Getting AI response...")
        ai_response = get_deepseek_response(user_input)
        logger.info(f"AI Response: {ai_response}")

        # Step 3: Text-to-Speech
        logger.info("Generating speech...")
        output_filename = f"output_{uuid.uuid4()}.mp3"
        output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)

        if tts_model:
            try:
                tts_model.tts_to_file(
                    text=ai_response,
                    speaker_wav=input_filepath,
                    file_path=output_filepath
                )
            except Exception as e:
                logger.error(f"XTTS failed, using gTTS: {str(e)}")
                tts = gTTS(text=ai_response, lang='tr')
                tts.save(output_filepath)
        else:
            tts = gTTS(text=ai_response, lang='tr')
            tts.save(output_filepath)

        # Add to chat history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'audio_url': f'/static/audio/{output_filename}'
        }
        chat_history.append(chat_entry)

        return jsonify({
            'status': 'success',
            'user_input': user_input,
            'response_text': ai_response,
            'audio_url': f'/static/audio/{output_filename}'
        })

    except Exception as e:
        logger.error(f"Error in ai_speak: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup input file
        try:
            os.remove(input_filepath)
        except Exception as e:
            logger.warning(f"Could not delete input file: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)