import torch
from transformers import AutoTokenizer, AutoModel

import numpy as np
from scipy.spatial.distance import cdist

# Load the model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute embeddings
def compute_embeddings(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    
    return embeddings

documents = [
    "The cat sits on the mat.",
    "Dogs are great pets.",
    "Birds can fly.",
    "I admire programmers",
    "I love programming in Python.",
    "Artificial Intelligence is fascinating."
]
document_embeddings = compute_embeddings(documents)


def compute_for_query(query):
    query_embedding = compute_embeddings([query])[0]
    cosine_similarities = 1 - cdist([query_embedding], document_embeddings, metric='cosine')[0]
    most_similar_index = np.argmax(cosine_similarities)
    return documents[most_similar_index], int(most_similar_index)


from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', documents=documents)

@app.route('/search', methods=['POST'])
def spam_check():
    input = request.json.get('input')
    print("Input: ", input)
    if input:
        result, index = compute_for_query(input)
        print("Result: ", result )
        return jsonify({'result': result, 'index': index}), 200
    else:
        return jsonify({'result': "Type something...", 'index': -1})

if __name__ == "__main__":
    app.run(port=2604)