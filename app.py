from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# --- Audio augmentation (if you want to support on‐the‐fly augment) ---
AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# --- Spectrogram parameters ---
SAMPLE_RATE = 16000
N_MELS = 64
FIXED_DURATION = 5  # seconds
SPEC_SHAPE = (64, 64)

def extract_mel_spectrogram(file_path):
    # load and optionally truncate/pad audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=FIXED_DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # pad or trim to fixed width
    if mel_db.shape[1] < SPEC_SHAPE[1]:
        pad_width = SPEC_SHAPE[1] - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    return mel_db[:, :SPEC_SHAPE[1]]

# --- Load your trained model (replace path as needed) ---
MODEL_PATH = 'cry_detector.h5'
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects multipart/form-data with:
    - file: the audio file (wav, mp3, etc.)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    audio = request.files['file']
    tmp_path = f"/tmp/{audio.filename}"
    audio.save(tmp_path)

    # extract features
    spec = extract_mel_spectrogram(tmp_path)
    x = spec[np.newaxis, ..., np.newaxis]  # (1,64,64,1)
    prob = float(model.predict(x)[0][0])
    label = int(prob > 0.5)

    return jsonify({'label': label, 'probability': prob})

if __name__ == '__main__':
    app.run(debug=True)
