from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Function to compute positional encoding
def positional_encoding(seq_length, embedding_dimension):
  # sin (pos / constant^{2i/embedding_dimension})
  positional_embeddings = []
  for pos in range(0, seq_length):
    sinusoidal_for_pos = []
    for i in range(0, embedding_dimension):
      sinusoidal_for_pos.append(np.sin(pos / (np.pow(2 * i/embedding_dimension, 2))))
    positional_embeddings.append(sinusoidal_for_pos)
  return positional_embeddings

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    seq_length = data.get('seq_length')
    d_model = data.get('d_model')
    encodings = positional_encoding(seq_length, d_model)
    return jsonify(encodings)

if __name__ == '__main__':
    app.run(debug=True)
