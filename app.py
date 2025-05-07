import os
import traceback
from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# added this to apply some changes to the audio so the model generalizes better
AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# just setting the basic constants that we’ll use while processing audio
SAMPLE_RATE    = 16000
N_MELS         = 64
FIXED_DURATION = 5
SPEC_SHAPE     = (64, 64)

# function to turn audio file into a mel spectrogram
def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=FIXED_DURATION)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # added this to make sure all spectrograms are same size
    if mel_db.shape[1] < SPEC_SHAPE[1]:
        pad_width = SPEC_SHAPE[1] - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    
    return mel_db[:, :SPEC_SHAPE[1]]

# grabbing current directory and loading the saved model from there
BASEDIR    = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASEDIR, 'cry_detector.h5')
model      = tf.keras.models.load_model(MODEL_PATH)

# used this to make predictions run a bit faster
infer = tf.function(lambda inp: model(inp, training=False))

# starting the flask app here
app = Flask(__name__)

# just a route to check if the server is working fine
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "ok", "message": "Cry-detector API is running!"})

# route to send audio and get prediction back
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # checking if file was actually sent
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        audio    = request.files['file']
        tmp_path = os.path.join('/tmp', audio.filename)
        audio.save(tmp_path)

        # just confirming that the file saved properly
        if not os.path.exists(tmp_path):
            return jsonify({'error': 'Failed to save upload'}), 500

        # extracting features from audio
        spec = extract_mel_spectrogram(tmp_path)
        x    = spec[np.newaxis, ..., np.newaxis].astype(np.float32)

        # running the prediction
        prob_tensor = infer(tf.constant(x))
        prob        = float(prob_tensor.numpy()[0][0])
        label       = int(prob > 0.5)

        return jsonify({'label': label, 'probability': prob})

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# running the flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
