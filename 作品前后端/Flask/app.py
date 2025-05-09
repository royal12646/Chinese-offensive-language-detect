from flask import Flask, request, jsonify
import torch
from utils import load_models_and_predict, Model, Model_COLD4, MultiHeadAttention 
from flask_cors import CORS
app = Flask(__name__, static_folder='static')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    model = data.get('model')
    print(text)
    print(model)
    prediction = load_models_and_predict(text, model, device)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)  # 关键！监听所有 IP 地址