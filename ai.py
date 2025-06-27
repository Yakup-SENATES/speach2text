from flask import Flask, request, jsonify, render_template
import whisper
from gtts import gTTS  # Import at the top
from TTS.api import TTS
import torch
import logging
import os
import uuid
import subprocess  # Ollama'yi çaliştirmak için
from flask_cors import CORS  # Import CORS

logging.basicConfig(level=logging.INFO)  # Configure logging
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Klasör yollari
UPLOAD_FOLDER = os.path.join(app.static_folder, 'audio')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Modelleri yükle
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading Whisper model... -> medium")
model = whisper.load_model("medium").to(device)

# List available 🐸TTS models
print(TTS().list_models())


# Ollama ile DeepSeek'e istek gönderme fonksiyonu
import subprocess
import logging

def get_deepseek_response(prompt: str) -> str | None:
    """
    DeepSeek dil modeline Ollama araciliğiyla metin gönderir ve yanit döndürür.

    Args:
        prompt (str): DeepSeek'e gönderilecek metin.

    Returns:
        str | None: DeepSeek'ten gelen yanit veya hata durumunda None.
    """
    try:
        logging.info(f"[DeepSeek] Giden İstek: {prompt}")

        command = ["ollama", "run", "mistral", prompt]
        process = subprocess.run(command, capture_output=True, text=True, check=True)

        if process.stdout:
            response = process.stdout.strip()
            logging.info(f"[DeepSeek] Gelen Yanit: {response}")
            return response

        logging.warning("[DeepSeek] Yanit boş döndü.")
        return None

    except subprocess.CalledProcessError as e:
        logging.error(f"[DeepSeek] Çaliştirma hatasi: {e.stderr}")
        return None
    except FileNotFoundError:
        logging.critical("[DeepSeek] Ollama bulunamadi! Ollama'nin yüklü ve PATH'e ekli olduğundan emin olun.")
        return None
    except Exception as e:
        logging.exception("[DeepSeek] Beklenmeyen hata oluştu:")
        return None




@app.route('/')
def home():
    return render_template('index2.html')


@app.route('/ai_speak', methods=['POST'])
def ai_speak():
    if 'audio' not in request.files:
        return jsonify({'error': 'Ses dosyasi yok'}), 400

    audio_file = request.files['audio']
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)
    print(f"Kaydedilen ses dosyasi: {filepath}")

    try:
        # 1. Kullanıcının sesini metne çevir (Whisper)
        print(f"Transcribing audio file: {filepath}")
        result = model.transcribe(filepath, task="transcribe")
        print(f"Whisper response: {result}")
        user_input = result["text"]

        # 2. AI modeline soruyu gönder
        logging.debug(f"Kullanici girdisi: {user_input}")
        deepseek_response = get_deepseek_response(user_input)
        print(f"DeepSeek yaniti: {deepseek_response}")

        if deepseek_response:
            # Metni sese dönüştür (gTTS)
            filename_tts = f"{uuid.uuid4()}.mp3"
            filepath_tts = os.path.join(UPLOAD_FOLDER, filename_tts)
            try:
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                print("XTTS modeli başarıyla yüklendi.")
            except Exception as e:
                print(f"XTTS modeli yüklenirken hata oluştu: {e}")
                tts = None # Modeli yükleyemezse None olarak işaretle

            if tts is None:
                # XTTS modeli yüklenemediyse gTTS kullan
                tts = gTTS(text=deepseek_response)
            else: 
                tts = tts.tts_to_file(
                    text=deepseek_response,
                    speaker_wav=filepath
                )



            tts = gTTS(text=deepseek_response)  # veya uygun dil
            tts.save(filepath_tts)

            # Ses URL'sini geri gönder
            return jsonify({'audio_url': f'/static/audio/{filename_tts}'})
        else:
            return jsonify({'error': 'DeepSeek yanit vermedi'}), 500

    except Exception as e:
        logging.exception("Hata:")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)