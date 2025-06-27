from flask import Flask, request, jsonify, render_template
import whisper
from transformers import pipeline
import torch
import os
import uuid
from gtts import gTTS  # Import at the top
import logging
from flask_cors import CORS  # Import CORS

logging.basicConfig(level=logging.ERROR)  # Configure logging
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

print("Loading translator model...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en", device=device)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']

    # Ses dosyasini kaydet
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    try:
        # Ses dosyasini metne çevir
        result = model.transcribe(filepath, language="tr", task="transcribe")
        original_text = result["text"]

        # Metni İngilizce'ye çevir
        translation = translator(original_text)[0]['translation_text']

        response_data = {
            'original_text': original_text,
            'translated_text': translation
        }

        # Temizlik
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(response_data)

    except Exception as e:
        logging.exception("Error in /transcribe:")
        return jsonify({'error': str(e)}), 500


@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text', '')
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Ses dosyasi oluştur
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        tts = gTTS(text=text.encode('utf-8').decode('utf-8'), lang=lang)  # UTF-8 encoding
        tts.save(filepath)

        return jsonify({
            'audio_url': f'/static/audio/{filename}'
        })

    except Exception as e:
        logging.exception("Error in /speak:")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)