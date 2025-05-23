from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import prediction_helpers as ph
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from umap import UMAP
import warnings
import re
warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app)

# Global variables for S3BERT
model = None
model_name = "s3bert_all-mpnet-base-v2"
feature_dim = 16
n_features = 15

# Global variable for topic similarity
topic_analyzer = None

# List of features for similarity scores
features = ["global"] + [
    'Concepts ', 'Frames ', 'Named Ent. ', 'Negations ', 'Reentrancies ',
    'SRL ', 'Smatch ', 'Unlabeled ', 'max_indegree_sim', 'max_outdegree_sim', 
    'max_degree_sim', 'root_sim', 'quant_sim', 'score_wlk', 'score_wwlk'
] + ["residual"]

class ArgumentTopicSimilarity:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the topic similarity analyzer

        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.topic_model = None

    def preprocess_arguments(self, arg1, arg2):
        """
        Preprocess arguments by splitting into sentences

        Args:
            arg1 (str): First argument text
            arg2 (str): Second argument text

        Returns:
            list: Combined list of sentences from both arguments
            dict: Mapping of sentence indices to argument source
        """
        def split_sentences(text):
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

        sentences_arg1 = split_sentences(arg1)
        sentences_arg2 = split_sentences(arg2)

        all_sentences = sentences_arg1 + sentences_arg2

        sentence_mapping = {}
        for i, _ in enumerate(sentences_arg1):
            sentence_mapping[i] = 'arg1'
        for i, _ in enumerate(sentences_arg2):
            sentence_mapping[i + len(sentences_arg1)] = 'arg2'

        return all_sentences, sentence_mapping

    def extract_topics(self, documents, min_topic_size=2):
        """
        Extract topics using BERTopic with fallback mechanisms for small datasets
        """
        n_docs = len(documents)

        if n_docs < 4:
            return self._simple_topic_assignment(documents)

        if min_topic_size >= n_docs:
            min_topic_size = max(2, n_docs // 2)

        hdbscan_model = HDBSCAN(
            min_cluster_size=max(2, min_topic_size),
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        umap_model = UMAP(
            n_neighbors=min(15, n_docs - 1),
            n_components=min(5, n_docs - 1),
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        vectorizer_model = CountVectorizer(stop_words="english", min_df=1, max_df=0.95)

        try:
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                hdbscan_model=hdbscan_model,
                umap_model=umap_model,
                vectorizer_model=vectorizer_model,
                verbose=False,
                calculate_probabilities=False
            )

            topics, probabilities = self.topic_model.fit_transform(documents)

            valid_topics = [t for t in topics if t != -1]

            if len(valid_topics) < 2:
                return self._fallback_clustering(documents)

        except Exception as e:
            return self._fallback_clustering(documents)

        return self.topic_model, topics

    def _simple_topic_assignment(self, documents):
        """Simple topic assignment for very small datasets"""
        embeddings = self.embedding_model.encode(documents)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1)

        n_clusters = min(2, len(documents))
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        try:
            cluster_labels = kmeans_model.fit_predict(embeddings)
            
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=vectorizer_model,
                verbose=False
            )

            topics = cluster_labels.tolist()

            unique_topics = list(set(topics))
            topic_info_data = []

            for topic_id in unique_topics:
                topic_docs = [documents[i] for i, t in enumerate(topics) if t == topic_id]
                topic_info_data.append({
                    'Topic': topic_id,
                    'Count': len(topic_docs),
                    'Name': f'Topic_{topic_id}'
                })

            self.topic_model._topic_info = pd.DataFrame(topic_info_data)

        except Exception as e:
            topics = [0] * len(documents)
            self.topic_model = type('MockModel', (), {
                'get_topic_info': lambda: pd.DataFrame({'Topic': [0], 'Count': [len(documents)], 'Name': ['Single_Topic']})
            })()

        return self.topic_model, topics

    def _fallback_clustering(self, documents):
        """Fallback clustering method using KMeans"""
        try:
            embeddings = self.embedding_model.encode(documents)
            n_clusters = min(max(2, len(documents) // 3), len(documents))
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

            cluster_labels = kmeans_model.fit_predict(embeddings)
            vectorizer_model = CountVectorizer(stop_words="english", min_df=1)

            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=vectorizer_model,
                verbose=False
            )

            topics = cluster_labels.tolist()

            try:
                doc_term_matrix = vectorizer_model.fit_transform(documents)
                unique_topics = list(set(topics))
                topic_info_data = []

                for topic_id in unique_topics:
                    topic_docs = [documents[i] for i, t in enumerate(topics) if t == topic_id]
                    topic_info_data.append({
                        'Topic': topic_id,
                        'Count': len(topic_docs),
                        'Name': f'Cluster_{topic_id}'
                    })

                self.topic_model._topic_info = pd.DataFrame(topic_info_data)

            except Exception as ve:
                self.topic_model._topic_info = pd.DataFrame({
                    'Topic': list(set(topics)),
                    'Count': [topics.count(t) for t in set(topics)],
                    'Name': [f'Topic_{t}' for t in set(topics)]
                })

            return self.topic_model, topics

        except Exception as e:
            topics = [0] * len(documents)

            class MockModel:
                def get_topic_info(self):
                    return pd.DataFrame({'Topic': [0], 'Count': [len(documents)], 'Name': ['All_Documents']})

                def get_topic(self, topic_id):
                    return [('word', 1.0) for _ in range(5)]

            self.topic_model = MockModel()
            return self.topic_model, topics

    def calculate_topic_distributions(self, topics, sentence_mapping):
        """Calculate topic distributions for each argument"""
        unique_topics = sorted([t for t in set(topics) if t != -1])

        if len(unique_topics) == 0:
            unique_topics = [0]
            topics = [0] * len(topics)

        arg1_topics = []
        arg2_topics = []

        for i, topic in enumerate(topics):
            if i < len(sentence_mapping):
                if sentence_mapping[i] == 'arg1':
                    arg1_topics.append(topic)
                else:
                    arg2_topics.append(topic)

        def get_distribution(topic_list, all_topics):
            if not topic_list or not all_topics:
                return np.zeros(max(1, len(all_topics)))

            distribution = np.zeros(len(all_topics))
            for topic in topic_list:
                if topic in all_topics:
                    idx = all_topics.index(topic)
                    distribution[idx] += 1
                elif topic == -1:
                    continue

            if distribution.sum() > 0:
                distribution = distribution / distribution.sum()
            else:
                distribution = np.ones(len(all_topics)) / len(all_topics)

            return distribution

        arg1_dist = get_distribution(arg1_topics, unique_topics)
        arg2_dist = get_distribution(arg2_topics, unique_topics)

        return {
            'arg1_distribution': arg1_dist,
            'arg2_distribution': arg2_dist,
            'topics': unique_topics,
            'arg1_topics': arg1_topics,
            'arg2_topics': arg2_topics
        }

    def calculate_similarity_metrics(self, arg1_dist, arg2_dist):
        """Calculate various similarity metrics between topic distributions"""
        if len(arg1_dist) != len(arg2_dist):
            max_len = max(len(arg1_dist), len(arg2_dist))
            arg1_dist = np.pad(arg1_dist, (0, max_len - len(arg1_dist)))
            arg2_dist = np.pad(arg2_dist, (0, max_len - len(arg2_dist)))

        if len(arg1_dist) == 0 or len(arg2_dist) == 0:
            return {
                'cosine_similarity': 0.0,
                'jaccard_similarity': 0.0,
                'kl_divergence': float('inf'),
                'js_divergence': 1.0,
                'js_distance': 1.0
            }

        try:
            cosine_sim = cosine_similarity([arg1_dist], [arg2_dist])[0][0]
            if np.isnan(cosine_sim):
                cosine_sim = 0.0
        except:
            cosine_sim = 0.0

        arg1_binary = (arg1_dist > 0).astype(int)
        arg2_binary = (arg2_dist > 0).astype(int)

        intersection = np.sum(arg1_binary & arg2_binary)
        union = np.sum(arg1_binary | arg2_binary)
        jaccard_sim = intersection / union if union > 0 else 0

        epsilon = 1e-10
        arg1_smooth = arg1_dist + epsilon
        arg2_smooth = arg2_dist + epsilon

        arg1_smooth = arg1_smooth / arg1_smooth.sum()
        arg2_smooth = arg2_smooth / arg2_smooth.sum()

        try:
            kl_div = np.sum(arg1_smooth * np.log(arg1_smooth / arg2_smooth))
            if np.isnan(kl_div) or np.isinf(kl_div):
                kl_div = float('inf')
        except:
            kl_div = float('inf')

        try:
            m = 0.5 * (arg1_smooth + arg2_smooth)
            js_div = 0.5 * np.sum(arg1_smooth * np.log(arg1_smooth / m)) + \
                     0.5 * np.sum(arg2_smooth * np.log(arg2_smooth / m))

            if np.isnan(js_div) or js_div < 0:
                js_div = 1.0

            js_distance = np.sqrt(js_div)
            if np.isnan(js_distance):
                js_distance = 1.0

        except:
            js_div = 1.0
            js_distance = 1.0

        return {
            'cosine_similarity': float(cosine_sim),
            'jaccard_similarity': float(jaccard_sim),
            'kl_divergence': float(kl_div),
            'js_divergence': float(js_div),
            'js_distance': float(js_distance)
        }

    def analyze_similarity(self, arg1, arg2, min_topic_size=2):
        """Complete pipeline to analyze topic similarity between two arguments"""
        sentences, mapping = self.preprocess_arguments(arg1, arg2)

        if len(sentences) < 2:
            return {
                'error': 'Not enough sentences to perform topic modeling',
                'similarity_score': 0.0
            }

        topic_model, topics = self.extract_topics(sentences, min_topic_size)
        distributions = self.calculate_topic_distributions(topics, mapping)
        similarities = self.calculate_similarity_metrics(
            distributions['arg1_distribution'],
            distributions['arg2_distribution']
        )

        try:
            if hasattr(topic_model, 'get_topic_info'):
                topic_info = topic_model.get_topic_info()
            elif hasattr(topic_model, '_topic_info'):
                topic_info = topic_model._topic_info
            else:
                unique_topics, counts = np.unique(topics, return_counts=True)
                topic_info = pd.DataFrame({
                    'Topic': unique_topics,
                    'Count': counts,
                    'Name': [f'Topic_{t}' for t in unique_topics]
                })
        except Exception as e:
            unique_topics, counts = np.unique(topics, return_counts=True)
            topic_info = pd.DataFrame({
                'Topic': unique_topics,
                'Count': counts,
                'Name': [f'Topic_{t}' for t in unique_topics]
            })

        return {
            'topic_model': topic_model,
            'topics': topics,
            'distributions': distributions,
            'similarities': similarities,
            'topic_info': topic_info,
            'sentences': sentences,
            'sentence_mapping': mapping
        }

def load_models():
    """Load both S3BERT and topic similarity models"""
    global model, topic_analyzer
    
    # Load S3BERT model
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    print(f"Loading S3BERT model {model_name} on {device}...")
    model = SentenceTransformer(f"./{model_name}/", device=device)
    print("S3BERT model loaded successfully")
    
    # Load topic similarity analyzer
    print("Loading topic similarity analyzer...")
    topic_analyzer = ArgumentTopicSimilarity()
    print("Topic similarity analyzer loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None or topic_analyzer is None:
        return jsonify({
            "status": "error",
            "message": "Models not loaded"
        }), 500
    
    return jsonify({
        "status": "ok",
        "s3bert_model": model_name,
        "topic_model": "ArgumentTopicSimilarity with all-MiniLM-L6-v2"
    })

# Original S3BERT endpoints
@app.route('/compare', methods=['POST'])
def compare_sentences():
    """Compare two sentences and return similarity scores"""
    data = request.get_json()
    
    if not data or 'argument1' not in data or 'argument2' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide both 'sentence1' and 'sentence2' in the request body"
        }), 400
    
    sentence1 = data['argument1']
    sentence2 = data['argument2']
    
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
        sent1_encoded = model.encode([sentence1])
        sent2_encoded = model.encode([sentence2])
        
        preds = ph.get_preds(
            sent1_encoded, 
            sent2_encoded, 
            n=n_features, 
            dim=feature_dim
        )
        
        result = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "overall_similarity": float(preds[0][0]),
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
    
    if not data or 'sentence_pairs' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'sentence_pairs' in the request body"
        }), 400
    
    sentence_pairs = data['sentence_pairs']
    
    if not isinstance(sentence_pairs, list):
        return jsonify({
            "status": "error",
            "message": "'sentence_pairs' must be a list of objects with 'sentence1' and 'sentence2'"
        }), 400
    
    sentence1_list = []
    sentence2_list = []
    valid_pairs = []
    
    for i, pair in enumerate(sentence_pairs):
        if not isinstance(pair, dict) or 'sentence1' not in pair or 'sentence2' not in pair:
            continue
            
        sentence1 = pair['argument1']
        sentence2 = pair['argument2']
        
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
        sent1_encoded = model.encode(sentence1_list)
        sent2_encoded = model.encode(sentence2_list)
        
        preds = ph.get_preds(
            sent1_encoded, 
            sent2_encoded, 
            n=n_features, 
            dim=feature_dim
        )
        
        results = []
        for i, (pair, pred) in enumerate(zip(valid_pairs, preds)):
            result = {
                "sentence1": pair['sentence1'],
                "sentence2": pair['sentence2'],
                "overall_similarity": float(pred[0]),
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

# New topic similarity endpoints
@app.route('/topic-similarity', methods=['POST'])
def analyze_topic_similarity():
    """Analyze topic similarity between two arguments"""
    data = request.get_json()
    
    if not data or 'argument1' not in data or 'argument2' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide both 'argument1' and 'argument2' in the request body"
        }), 400
    
    argument1 = data['argument1']
    argument2 = data['argument2']
    min_topic_size = data.get('min_topic_size', 2)
    
    if not isinstance(argument1, str) or not isinstance(argument2, str):
        return jsonify({
            "status": "error", 
            "message": "Both arguments must be strings"
        }), 400
    
    if not argument1.strip() or not argument2.strip():
        return jsonify({
            "status": "error",
            "message": "Arguments cannot be empty"
        }), 400
    
    try:
        results = topic_analyzer.analyze_similarity(
            argument1, 
            argument2, 
            min_topic_size=min_topic_size
        )
        
        if 'error' in results:
            return jsonify({
                "status": "error",
                "message": results['error']
            }), 400
        
        # Format response
        similarities = results['similarities']
        topic_info = results['topic_info'].to_dict('records') if hasattr(results['topic_info'], 'to_dict') else []
        
        # Overall similarity interpretation
        overall_sim = similarities['cosine_similarity']
        if overall_sim > 0.8:
            interpretation = "Very High Similarity"
        elif overall_sim > 0.6:
            interpretation = "High Similarity"
        elif overall_sim > 0.4:
            interpretation = "Moderate Similarity"
        elif overall_sim > 0.2:
            interpretation = "Low Similarity"
        else:
            interpretation = "Very Low Similarity"
        
        response_data = {
            "argument1": argument1,
            "argument2": argument2,
            "similarities": {
                "cosine_similarity": similarities['cosine_similarity'],
                "jaccard_similarity": similarities['jaccard_similarity'],
                "js_distance": similarities['js_distance']
            },
            "interpretation": interpretation,
            "topic_info": {
                "total_sentences": len(results['sentences']),
                "total_topics": len([t for t in results['topics'] if t != -1]),
                "topics": topic_info
            },
            "distributions": {
                "argument1_topics": results['distributions']['arg1_topics'],
                "argument2_topics": results['distributions']['arg2_topics']
            }
        }
        
        return jsonify({
            "status": "success",
            "result": response_data
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/batch-topic-similarity', methods=['POST'])
def batch_analyze_topic_similarity():
    """Analyze topic similarity for multiple argument pairs"""
    data = request.get_json()
    
    if not data or 'argument_pairs' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'argument_pairs' in the request body"
        }), 400
    
    argument_pairs = data['argument_pairs']
    min_topic_size = data.get('min_topic_size', 2)
    
    if not isinstance(argument_pairs, list):
        return jsonify({
            "status": "error",
            "message": "'argument_pairs' must be a list of objects with 'argument1' and 'argument2'"
        }), 400
    
    valid_pairs = []
    for pair in argument_pairs:
        if (isinstance(pair, dict) and 
            'argument1' in pair and 'argument2' in pair and
            isinstance(pair['argument1'], str) and isinstance(pair['argument2'], str) and
            pair['argument1'].strip() and pair['argument2'].strip()):
            valid_pairs.append(pair)
    
    if not valid_pairs:
        return jsonify({
            "status": "error",
            "message": "No valid argument pairs provided"
        }), 400
    
    try:
        results = []
        for pair in valid_pairs:
            analysis = topic_analyzer.analyze_similarity(
                pair['argument1'], 
                pair['argument2'], 
                min_topic_size=min_topic_size
            )
            
            if 'error' not in analysis:
                similarities = analysis['similarities']
                topic_info = analysis['topic_info'].to_dict('records') if hasattr(analysis['topic_info'], 'to_dict') else []
                
                overall_sim = similarities['cosine_similarity']
                if overall_sim > 0.8:
                    interpretation = "Very High Similarity"
                elif overall_sim > 0.6:
                    interpretation = "High Similarity"
                elif overall_sim > 0.4:
                    interpretation = "Moderate Similarity"
                elif overall_sim > 0.2:
                    interpretation = "Low Similarity"
                else:
                    interpretation = "Very Low Similarity"
                
                result = {
                    "argument1": pair['argument1'],
                    "argument2": pair['argument2'],
                    "similarities": {
                        "cosine_similarity": similarities['cosine_similarity'],
                        "jaccard_similarity": similarities['jaccard_similarity'],
                        "js_distance": similarities['js_distance']
                    },
                    "interpretation": interpretation,
                    "topic_info": {
                        "total_sentences": len(analysis['sentences']),
                        "total_topics": len([t for t in analysis['topics'] if t != -1]),
                        "topics": topic_info
                    }
                }
                results.append(result)
            else:
                results.append({
                    "argument1": pair['argument1'],
                    "argument2": pair['argument2'],
                    "error": analysis['error']
                })
        
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
    """Get the list of semantic features analyzed by the S3BERT model"""
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

@app.route('/endpoints', methods=['GET'])
def get_endpoints():
    """Get information about available API endpoints"""
    return jsonify({
        "status": "success",
        "endpoints": {
            "sentence_similarity": {
                "/compare": "Compare two sentences using S3BERT",
                "/batch-compare": "Compare multiple sentence pairs using S3BERT",
                "/features": "Get list of semantic features analyzed by S3BERT"
            },
            "topic_similarity": {
                "/topic-similarity": "Analyze topic similarity between two arguments",
                "/batch-topic-similarity": "Analyze topic similarity for multiple argument pairs"
            },
            "utility": {
                "/health": "Health check endpoint",
                "/endpoints": "This endpoint - lists all available endpoints"
            }
        }
    })

if __name__ == '__main__':
    # Load models at startup
    load_models()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)