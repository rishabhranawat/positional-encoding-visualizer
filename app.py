from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    # Tokenize input text and get tensor
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the last hidden state
    embeddings = outputs.last_hidden_state
    # Convert to numpy array and mean pool over tokens for simplicity
    print(embeddings.shape)
    embeddings = embeddings.mean(dim=0).numpy()
    print(embeddings.shape)
    return embeddings

def positional_encoding(seq_length, embedding_dimension):
    positional_embeddings = []
    for pos in range(seq_length):
        sinusoidal_for_pos = []
        for i in range(1, embedding_dimension + 1):
            sinusoidal_for_pos.append(np.sin(pos / (math.pow(2 * i / embedding_dimension, 2))))
        positional_embeddings.append(sinusoidal_for_pos)
    return positional_embeddings

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    text = data.get('text')

    # Tokenize text and compute BERT embeddings
    embeddings = get_bert_embeddings(text)

    d_model = embeddings.shape[1]  # Embedding dimension from BERT model
    seq_length = embeddings.shape[0]  # Sequence length after tokenization and padding/truncation

    # Compute positional encodings
    pos_encodings = positional_encoding(seq_length, d_model)
    print(np.array(pos_encodings).shape)

    # Combine embeddings with positional encodings
    combined = embeddings + np.array(pos_encodings)
    
    return jsonify({
        'tokens': tokenizer.tokenize(text),
        'embeddings': embeddings.tolist(),
        'positional_encodings': pos_encodings,
        'combined': combined.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
