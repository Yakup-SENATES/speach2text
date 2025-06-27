from flask import Flask, request, jsonify, render_template
import torch
import logging
import os
import uuid
import requests  # API istekleri için

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(app.static_folder, 'audio')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/deepseek_speak', methods=['POST'])
def deepseek_speak():
    if 'audio' not in request.files:
        return jsonify({'error': 'Ses dosyasi yok'}), 400

    audio_file = request.files['audio']
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    try:
        # 1. Kullanıcının sesini metne çevir (Whisper)
        result = model.transcribe(filepath, task="transcribe")
        user_input = result["text"]

        # 2. API.yasa.ai'ye istek gönder
        api_url = "https://api.yasa.ai/v1/ask"
        headers = {
            "Authorization": "YOUR_API_KEY"
        }
        data = {
            "question": user_input
        }
        response = requests.post(api_url, headers=headers, json=data)

        if response.status_code == 200:
            deepseek_response = response.json().get("answer")
        else:
            return jsonify({'error': 'API yanit vermedi'}), 500

        # 3. Metni sese dönüştür (gTTS)
        filename_tts = f"{uuid.uuid4()}.mp3"
        filepath_tts = os.path.join(UPLOAD_FOLDER, filename_tts)
        tts = gTTS(text=deepseek_response)
        tts.save(filepath_tts)

        return jsonify({'audio_url': f'/static/audio/{filename_tts}'})
    except Exception as e:
        logging.exception("Hata:")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)