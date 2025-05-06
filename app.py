import os
import traceback
from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# --- Audio augmentation (optional) ---
AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# --- Spectrogram parameters ---
SAMPLE_RATE    = 16000
N_MELS         = 64
FIXED_DURATION = 5    # seconds
SPEC_SHAPE     = (64, 64)

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=FIXED_DURATION)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # pad or trim to SPEC_SHAPE[1]
    if mel_db.shape[1] < SPEC_SHAPE[1]:
        pad_width = SPEC_SHAPE[1] - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    return mel_db[:, :SPEC_SHAPE[1]]

# --- Locate and load the model ---
BASEDIR    = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASEDIR, 'cry_detector.h5')
model      = tf.keras.models.load_model(MODEL_PATH)

# --- Compile a tf.function for faster inference ---
infer = tf.function(lambda inp: model(inp, training=False))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "ok", "message": "Cry-detector API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        audio    = request.files['file']
        tmp_path = os.path.join('/tmp', audio.filename)
        audio.save(tmp_path)

        if not os.path.exists(tmp_path):
            return jsonify({'error': 'Failed to save upload'}), 500

        spec = extract_mel_spectrogram(tmp_path)
        x    = spec[np.newaxis, ..., np.newaxis].astype(np.float32)  # shape (1,64,64,1)

        # fast inference
        prob_tensor = infer(tf.constant(x))
        prob        = float(prob_tensor.numpy()[0][0])
        label       = int(prob > 0.5)

        return jsonify({'label': label, 'probability': prob})

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
