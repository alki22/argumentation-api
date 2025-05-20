from flask import Flask, request, jsonify
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import prediction_helpers as ph

app = Flask(__name__)

# Global variables
model = None
model_name = "s3bert_all-mpnet-base-v2"
feature_dim = 16
n_features = 15

# List of features for similarity scores
features = ["global"] + [
    'Concepts ', 'Frames ', 'Named Ent. ', 'Negations ', 'Reentrancies ',
    'SRL ', 'Smatch ', 'Unlabeled ', 'max_indegree_sim', 'max_outdegree_sim', 
    'max_degree_sim', 'root_sim', 'quant_sim', 'score_wlk', 'score_wwlk'
] + ["residual"]

def load_model():
    """Load the S3BERT model"""
    global model
    
    # Check for GPU availability
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    model = SentenceTransformer(f"./{model_name}/", device=device)
    print("Model loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 500
    
    return jsonify({
        "status": "ok",
        "model": model_name
    })

@app.route('/compare', methods=['POST'])
def compare_sentences():
    """Compare two sentences and return similarity scores"""
    data = request.get_json()
    
    # Validate input
    if not data or 'sentence1' not in data or 'sentence2' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide both 'sentence1' and 'sentence2' in the request body"
        }), 400
    
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']
    
    # Validate sentences
    if not isinstance(sentence1, str) or not isinstance(sentence2, str):
        return jsonify({
            "status": "error", 
            "message": "Both sentences must be strings"
        }), 400
    
    if not sentence1.strip() or not sentence2.strip():
        return jsonify({
            "status": "error",
            "message": "Sentences cannot be empty"
        }), 400
    
    try:
        # Encode sentences
        sent1_encoded = model.encode([sentence1])
        sent2_encoded = model.encode([sentence2])
        
        # Get feature-wise similarity predictions
        preds = ph.get_preds(
            sent1_encoded, 
            sent2_encoded, 
            n=n_features, 
            dim=feature_dim
        )
        
        # Format results
        result = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "overall_similarity": float(preds[0][0]),  # Global similarity
            "feature_similarities": {
                feature: float(score) for feature, score in zip(features, preds[0])
            }
        }
        
        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/batch-compare', methods=['POST'])
def batch_compare_sentences():
    """Compare multiple sentence pairs and return similarity scores"""
    data = request.get_json()
    
    # Validate input
    if not data or 'sentence_pairs' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'sentence_pairs' in the request body"
        }), 400
    
    sentence_pairs = data['sentence_pairs']
    
    # Validate pairs
    if not isinstance(sentence_pairs, list):
        return jsonify({
            "status": "error",
            "message": "'sentence_pairs' must be a list of objects with 'sentence1' and 'sentence2'"
        }), 400
    
    # Extract sentence pairs
    sentence1_list = []
    sentence2_list = []
    valid_pairs = []
    
    for i, pair in enumerate(sentence_pairs):
        if not isinstance(pair, dict) or 'sentence1' not in pair or 'sentence2' not in pair:
            continue
            
        sentence1 = pair['sentence1']
        sentence2 = pair['sentence2']
        
        if not isinstance(sentence1, str) or not isinstance(sentence2, str):
            continue
            
        if not sentence1.strip() or not sentence2.strip():
            continue
            
        sentence1_list.append(sentence1)
        sentence2_list.append(sentence2)
        valid_pairs.append(pair)
    
    if not valid_pairs:
        return jsonify({
            "status": "error",
            "message": "No valid sentence pairs provided"
        }), 400
    
    try:
        # Encode sentences
        sent1_encoded = model.encode(sentence1_list)
        sent2_encoded = model.encode(sentence2_list)
        
        # Get feature-wise similarity predictions
        preds = ph.get_preds(
            sent1_encoded, 
            sent2_encoded, 
            n=n_features, 
            dim=feature_dim
        )
        
        # Format results
        results = []
        for i, (pair, pred) in enumerate(zip(valid_pairs, preds)):
            result = {
                "sentence1": pair['sentence1'],
                "sentence2": pair['sentence2'],
                "overall_similarity": float(pred[0]),  # Global similarity
                "feature_similarities": {
                    feature: float(score) for feature, score in zip(features, pred)
                }
            }
            results.append(result)
        
        return jsonify({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get the list of semantic features analyzed by the model"""
    return jsonify({
        "status": "success",
        "features": features,
        "description": {
            "global": "Overall sentence similarity",
            "Concepts ": "Similarity with respect to concepts in sentences",
            "Frames ": "Similarity with respect to predicates in sentences",
            "Named Ent. ": "Similarity with respect to named entities in sentences",
            "Negations ": "Similarity with respect to negation structure of sentences",
            "Reentrancies ": "Similarity with respect to coreference structure of sentences",
            "SRL ": "Similarity with respect to semantic role structure of sentences",
            "Smatch ": "Similarity with respect to overall semantic meaning structures",
            "Unlabeled ": "Similarity with respect to semantic meaning structures minus relation labels",
            "max_indegree_sim": "Similarity with respect to connected nodes (in-degree) in meaning space",
            "max_outdegree_sim": "Similarity with respect to connected nodes (out-degree) in meaning space",
            "max_degree_sim": "Similarity with respect to connected nodes (degree) in meaning space",
            "root_sim": "Similarity with respect to root nodes in semantic graphs",
            "quant_sim": "Similarity with respect to quantificational structure",
            "score_wlk": "Similarity measured with contextual Weisfeiler Leman Kernel",
            "score_wwlk": "Similarity measured with Wasserstein Weisfeiler Leman Kernel",
            "residual": "Residual similarity information not captured by other features"
        }
    })

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)