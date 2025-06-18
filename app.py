from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from google.api_core import exceptions
from pydantic import BaseModel, Field
from google import genai
import json
import time
import enum
import random
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
#cors = CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, 
      origins="*",
      methods=["GET", "POST", "OPTIONS"],
      allow_headers=["Content-Type", "Authorization"]
    )

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
# Global variable for stance classification
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL_ID = "gemini-2.0-flash"

stance_classification_prompt = """
You are a stance detection system. Given an argument and a topic, determine the stance (position) that the argument takes toward the topic.

Task: Analyze the provided argument and determine its stance toward the given topic, then provide a clear justification for your classification.

Output Format: Return one of the following three stance labels with justification:
- FAVOR: The argument supports or is in favor of the topic
- AGAINST: The argument opposes or is against the topic  
- NEUTRAL: The argument neither clearly supports nor opposes the topic, or presents a balanced view

Instructions:
1. Read the argument carefully
2. Identify key phrases, sentiments, and positions expressed
3. Determine how these relate to the given topic
4. Classify the overall stance
5. Provide a brief justification explaining your reasoning

Input Format:
Topic: [TOPIC]
Argument: [ARGUMENT]

Output Format:
Stance: [FAVOR/AGAINST/NEUTRAL]
Justification: [Explain the key elements in the argument that led to this classification, citing specific phrases or reasoning patterns]

Example:
Topic: Electric vehicles
Argument: "Electric cars are expensive and have limited range, making them impractical for most families."
Stance: AGAINST
Justification: The argument presents two negative characteristics of electric vehicles ("expensive" and "limited range") and concludes they are "impractical for most families," clearly opposing their adoption.

Your Task:
Topic: {topic}
Argument: {argument}
Stance:
Justification:"""

premise_claim_detection_prompt = """
You are an expert in argument mining and logic. Your task is to analyze an argument by extracting its premises and claim, while preserving the original wording as much as possible. 
Task:
Given a text containing an argument, you must:
1. Identify and extract the premise(s) that logically support the claim.
2. Identify and extract the claim that the argument is trying to establish.
3. If a premise or claim is implicit or missing, return an empty value for that component.

Additional Constraints:

1. Keep the wording as close as possible to the original sentence.
2. If either the premise or claim is implicit, return an empty value for that field.
3. If there's not a clear premise-claim structure or either of them is not clear, just return the original sentence (without paraphrasing) as both premise and claim

Input Format:
Argument: [ARGUMENT]

Output Format:
Premise: [EXTRACTED PREMISE]
Claim: [EXTRACTED CLAIM]

Your Task:
Argument: {argument}
Premise:
Claim:
"""

reasoning_type_prompt = """
You are an expert in argument mining and logic. Your task is to analyze the type of reasoning used in the argument based on the following definitions:

Deductive Reasoning:
Deductive reasoning moves from general premises to a specific conclusion that logically follows. If the premises are true, the conclusion must also be true. It is characterized by necessity and validity.

Inductive Reasoning:
Inductive reasoning moves from specific observations to a general conclusion. The conclusion is probable but not guaranteed.

Abductive Reasoning:
Abductive reasoning infers the most likely explanation from incomplete evidence. It is commonly used in diagnostics and hypothesis formation.

Analogical Reasoning:
Analogical reasoning draws a conclusion based on the similarity between two situations. The strength of the argument depends on how relevant the analogy is.

Instructions:
1. Read the argument carefully
2. Identify relations between claims and premises
3. Determine how these relate to each other
4. Classify the reasoning type
5. Provide a brief justification explaining your reasoning

Input Format:
Argument: [ARGUMENT]

Output Format:
Stance: [DEDUCTIVE/INDUCTIVE/ABDUCTIVE/ANALOGICAL]
Justification: [Explain the key elements in the argument that led to this classification, citing specific phrases or reasoning patterns]

Your Task:
Argument: {argument}
Reasoning type:
Justification:
"""

topic_detection_prompt = """
    You are a topic detection system. Given an argument, determine the most relevant topics that the argument discusses.
    Task: Analyze the provided argument and determine its relevant topics
    Output Format: Return a list with the two most relevant topics topics
    Instructions:
    1. Read the argument carefully
    2. Identify key phrases, sentiments, and positions expressed
    3. Provide a list of two topics, ordered from most to least relevant
    Your Task:
    Argument: {argument}
"""

class Stance(enum.Enum):
    FOR = "For"
    AGAINST = "Against"
    NEUTRAL = "Neutral"

class StanceClassification(BaseModel):
    stance: Stance = Field(description="The type of reasoning used in the argument")
    justification: str = Field(description="The justification to support the provided stance")


def annotate_stance(argument, topic):
    retries = 0
    delay = 5

    while retries < 20:
        try:
            full_prompt = stance_classification_prompt.format(argument=argument, topic=topic)
            response = llm_client.models.generate_content(model=GEMINI_MODEL_ID,
                                                      contents=full_prompt,
                                                      config={
                                                        'response_mime_type': 'application/json',
                                                        'response_schema': StanceClassification,
                                                        'temperature' : 0.2,
                                                        }
                                                      )
            #time.sleep(5)
            return json.loads(response.candidates[0].content.parts[0].text.strip())
        except exceptions.ResourceExhausted as e:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay + random.uniform(0, 1))  # Add some jitter
            delay *= 2  # Exponential backoff
            retries += 1
        except Exception as e:
            print(f"Error processing text: {argument}\nError: {str(e)}")
            return None

    print(f"Max retries reached for text: {argument}")
    return None

class Reasoning(enum.Enum):
    DEDUCTIVE = "Deductive"
    INDUCTIVE = "Inductive"
    ABDUCTIVE = "Abductive"
    ANALOGICAL = "Analogical"


class ReasoningTypeAnnotation(BaseModel):
    reasoning_type: Reasoning = Field(description="The type of reasoning used in the argument")
    justification: str = Field(description="Justification for the chosen reasoning type")

def annotate_reasoning_type(argument):
    """Send text to Gemini API for reasoning type annotation"""
    retries = 0
    delay = 5

    while retries < 20:
        try:
            full_prompt = reasoning_type_prompt.format(argument=argument)
            response = llm_client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=full_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ReasoningTypeAnnotation,
                    'temperature': 0.2,
                }
            )
            time.sleep(1)  # Reduced sleep time
            return json.loads(response.candidates[0].content.parts[0].text.strip())
        except exceptions.ResourceExhausted as e:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay + random.uniform(0, 1))  # Add some jitter
            delay *= 2  # Exponential backoff
            retries += 1
        except Exception as e:
            print(f"Error processing text: {argument}\nError: {str(e)}")
            return None

    print(f"Max retries reached for text: {argument}")
    return None

class TopicDetection(BaseModel):
    topics: list[str] = Field(description="The topics that are present in the argument")

def annotate_topics(argument):
    """Send text to Gemini API for topic annotation"""
    retries = 0
    delay = 5

    while retries < 20:
        try:
            full_prompt = topic_detection_prompt.format(argument=argument)
            response = llm_client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=full_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': TopicDetection,
                    'temperature': 0.2,
                }
            )
            time.sleep(1)  # Reduced sleep time
            return json.loads(response.candidates[0].content.parts[0].text.strip())
        except exceptions.ResourceExhausted as e:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay + random.uniform(0, 1))  # Add some jitter
            delay *= 2  # Exponential backoff
            retries += 1
        except Exception as e:
            print(f"Error processing text: {argument}\nError: {str(e)}")
            return None

    print(f"Max retries reached for text: {argument}")
    return None



class PremiseClaimDetection(BaseModel):
    premise: str = Field(description="The argument extracted premise")
    claim: str = Field(description="The argument extracted claim")

def annotate_premise_claim(argument):
    """Send text to Gemini API for premise-claim extraction"""
    retries = 0
    delay = 5

    while retries < 20:
        try:
            full_prompt = premise_claim_detection_prompt.format(argument=argument)
            response = llm_client.models.generate_content(model=GEMINI_MODEL_ID,  # Fixed: was using 'client' instead of 'llm_client'
                                                      contents=full_prompt,
                                                      config={
                                                        'response_mime_type': 'application/json',
                                                        'response_schema': PremiseClaimDetection,
                                                        'temperature' : 0.2,
                                                        }
                                                      )
            time.sleep(1)  # Reduced sleep time to match other functions
            return json.loads(response.candidates[0].content.parts[0].text.strip())
        except exceptions.ResourceExhausted as e:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay + random.uniform(0, 1))  # Add some jitter
            delay *= 2  # Exponential backoff
            retries += 1
        except Exception as e:
            print(f"Error processing text: {argument}\nError: {str(e)}")
            return None

    print(f"Max retries reached for text: {argument}")
    return None

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
    
    def calculate_llm_topic_similarity(self, arg1, arg2):
        """
        Calculate topic similarity using LLM-generated topics and S3BERT embeddings
        
        Args:
            arg1 (str): First argument text
            arg2 (str): Second argument text
        
        Returns:
            dict: Results containing top similarity score and most similar topic pairs
        """
        try:
            # Step 1: Get topics for both arguments using LLM
            topics_arg1_result = annotate_topics(arg1)
            topics_arg2_result = annotate_topics(arg2)
            
            if not topics_arg1_result or not topics_arg2_result:
                return {
                    'error': 'Failed to extract topics from one or both arguments',
                    'top_similarity': 0.0,
                    'top_pairs': []
                }
            
            topics_arg1 = topics_arg1_result.get('topics', [])
            topics_arg2 = topics_arg2_result.get('topics', [])
            
            if not topics_arg1 or not topics_arg2:
                return {
                    'error': 'No topics found in one or both arguments',
                    'top_similarity': 0.0,
                    'top_pairs': []
                }
            
            # Step 2: Calculate S3BERT embeddings for all topics
            all_topics = topics_arg1 + topics_arg2
            topic_embeddings = model.encode(all_topics)
            
            # Split embeddings back into separate lists
            embeddings_arg1 = topic_embeddings[:len(topics_arg1)]
            embeddings_arg2 = topic_embeddings[len(topics_arg1):]
            
            # Step 3: Calculate cosine similarity between all pairs
            similarity_results = []
            
            for i, (topic1, emb1) in enumerate(zip(topics_arg1, embeddings_arg1)):
                for j, (topic2, emb2) in enumerate(zip(topics_arg2, embeddings_arg2)):
                    # Calculate cosine similarity
                    similarity = cosine_similarity([emb1], [emb2])[0][0]
                    
                    similarity_results.append({
                        'topic1': topic1,
                        'topic2': topic2,
                        'similarity': float(similarity),
                        'topic1_index': i,
                        'topic2_index': j
                    })
            
            # Step 4: Sort by similarity and get top results
            similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Get top similarity score
            top_similarity = similarity_results[0]['similarity'] if similarity_results else 0.0
            
            # Get top 3 most similar pairs (topics only, not similarity values)
            top_pairs = [
                {
                    'topic1': result['topic1'],
                    'topic2': result['topic2'],
                    'similarity': result['similarity']
                }
                for result in similarity_results[:3]
            ]
            
            return {
                'topics_arg1': topics_arg1,
                'topics_arg2': topics_arg2,
                'top_similarity': top_similarity,
                'top_pairs': top_pairs,
                'all_similarities': similarity_results
            }
            
        except Exception as e:
            return {
                'error': f'Error calculating LLM topic similarity: {str(e)}',
                'top_similarity': 0.0,
                'top_pairs': []
            }

def load_models():
    """Load both S3BERT and topic similarity models"""
    global model, topic_analyzer, llm_client
    
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
    
    print("Loading stance classification model...")
    llm_client = genai.Client(api_key=GEMINI_API_KEY)
    print("Stance classification model loaded successfully")
    

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
@app.route('/compare', methods=['POST', 'OPTIONS'])
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

@app.route('/batch-compare', methods=['POST', 'OPTIONS'])
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
@app.route('/topic-similarity', methods=['POST', 'OPTIONS'])
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

@app.route('/batch-topic-similarity', methods=['POST', 'OPTIONS'])
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


@app.route('/extract-topics', methods=['POST', 'OPTIONS'])
def extract_topics():
    """Extract topics from a single argument using LLM"""
    data = request.get_json()
    
    if not data or 'argument' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'argument' in the request body"
        }), 400
    
    argument = data['argument']
    
    if not isinstance(argument, str):
        return jsonify({
            "status": "error", 
            "message": "Argument must be a string"
        }), 400
    
    if not argument.strip():
        return jsonify({
            "status": "error",
            "message": "Argument cannot be empty"
        }), 400
    
    try:
        # Use the existing annotate_topics function
        result = annotate_topics(argument)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": "Failed to extract topics from the argument"
            }), 500
        
        topics = result.get('topics', [])
        
        if not topics:
            return jsonify({
                "status": "error",
                "message": "No topics found in the argument"
            }), 400
        
        response_data = {
            "argument": argument,
            "topics": topics,
            "topic_count": len(topics)
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

@app.route('/batch-extract-topics', methods=['POST', 'OPTIONS'])
def batch_extract_topics():
    """Extract topics from multiple arguments using LLM"""
    data = request.get_json()
    
    if not data or 'arguments' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'arguments' list in the request body"
        }), 400
    
    arguments = data['arguments']
    
    if not isinstance(arguments, list):
        return jsonify({
            "status": "error",
            "message": "'arguments' must be a list of strings"
        }), 400
    
    # Filter valid arguments
    valid_arguments = []
    for arg in arguments:
        if isinstance(arg, str) and arg.strip():
            valid_arguments.append(arg)
    
    if not valid_arguments:
        return jsonify({
            "status": "error",
            "message": "No valid arguments provided"
        }), 400
    
    try:
        results = []
        for argument in valid_arguments:
            # Use the existing annotate_topics function
            topic_result = annotate_topics(argument)
            
            if topic_result:
                topics = topic_result.get('topics', [])
                result = {
                    "argument": argument,
                    "topics": topics,
                    "topic_count": len(topics),
                    "status": "success"
                }
            else:
                result = {
                    "argument": argument,
                    "topics": [],
                    "topic_count": 0,
                    "status": "error",
                    "error": "Failed to extract topics"
                }
            
            results.append(result)
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/topic-similarity-llm', methods=['POST', 'OPTIONS'])
def analyze_llm_topic_similarity():
    """Analyze topic similarity using LLM-generated topics and S3BERT embeddings"""
    data = request.get_json()
    
    if not data or 'argument1' not in data or 'argument2' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide both 'argument1' and 'argument2' in the request body"
        }), 400
    
    argument1 = data['argument1']
    argument2 = data['argument2']
    
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
        results = topic_analyzer.calculate_llm_topic_similarity(argument1, argument2)
        
        if 'error' in results:
            return jsonify({
                "status": "error",
                "message": results['error']
            }), 400
        
        # Format response
        top_similarity = results['top_similarity']
        
        # Similarity interpretation
        if top_similarity > 0.8:
            interpretation = "Very High Topic Similarity"
        elif top_similarity > 0.6:
            interpretation = "High Topic Similarity"
        elif top_similarity > 0.4:
            interpretation = "Moderate Topic Similarity"
        elif top_similarity > 0.2:
            interpretation = "Low Topic Similarity"
        else:
            interpretation = "Very Low Topic Similarity"
        
        response_data = {
            "argument1": argument1,
            "argument2": argument2,
            "topics_argument1": results['topics_arg1'],
            "topics_argument2": results['topics_arg2'],
            "top_similarity_score": top_similarity,
            "interpretation": interpretation,
            "top_similar_pairs": [
                {
                    "topic_from_arg1": pair['topic1'],
                    "topic_from_arg2": pair['topic2'],
                    "similarity_score": pair['similarity']
                }
                for pair in results['top_pairs']
            ],
            "total_comparisons": len(results['all_similarities'])
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


@app.route('/stance-classification', methods=['POST', 'OPTIONS'])
def classify_stance():
    """Given an argument and a topic, determine the stance (position) that the argument takes toward the topic."""
    data = request.get_json()

    if not data or 'argument1' not in data or 'argument2' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide both 'sentence1' and 'sentence2' in the request body"
        }), 400

    argument = data['argument1']
    topic = data['argument2']

    if not isinstance(argument, str) or not isinstance(topic, str):
        return jsonify({
            "status": "error", 
            "message": "The argument and topic must be strings"
        }), 400

    if not argument.strip() or not topic.strip():
        return jsonify({
            "status": "error",
            "message": "The argument and topic cannot be empty"
        }), 400

    try:

        result = annotate_stance(argument, topic)

        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/reasoning-type-classification', methods=['POST', 'OPTIONS'])
def classify_reasoning_type():
    data = request.get_json()

    if not data or 'argument1' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide both 'sentence1' and 'sentence2' in the request body"
        }), 400

    argument = data['argument1']

    if not isinstance(argument, str):
        return jsonify({
            "status": "error", 
            "message": "The argument must be a string"
        }), 400

    if not argument.strip():
        return jsonify({
            "status": "error",
            "message": "The argument cannot be empty"
        }), 400

    try:

        result = annotate_reasoning_type(argument)

        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/extract-premise-claim', methods=['POST', 'OPTIONS'])
def extract_premise_claim():
    """Extract premise and claim from a single argument using LLM"""
    data = request.get_json()
    
    if not data or 'argument' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'argument' in the request body"
        }), 400
    
    argument = data['argument']
    
    if not isinstance(argument, str):
        return jsonify({
            "status": "error", 
            "message": "Argument must be a string"
        }), 400
    
    if not argument.strip():
        return jsonify({
            "status": "error",
            "message": "Argument cannot be empty"
        }), 400
    
    try:
        # Use the existing annotate_premise_claim function
        result = annotate_premise_claim(argument)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": "Failed to extract premise and claim from the argument"
            }), 500
        
        premise = result.get('premise', '')
        claim = result.get('claim', '')
        
        response_data = {
            "argument": argument,
            "premise": premise,
            "claim": claim,
            "has_premise": bool(premise and premise.strip()),
            "has_claim": bool(claim and claim.strip())
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

@app.route('/batch-extract-premise-claim', methods=['POST', 'OPTIONS'])
def batch_extract_premise_claim():
    """Extract premise and claim from multiple arguments using LLM"""
    data = request.get_json()
    
    if not data or 'arguments' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'arguments' list in the request body"
        }), 400
    
    arguments = data['arguments']
    
    if not isinstance(arguments, list):
        return jsonify({
            "status": "error",
            "message": "'arguments' must be a list of strings"
        }), 400
    
    # Filter valid arguments
    valid_arguments = []
    for arg in arguments:
        if isinstance(arg, str) and arg.strip():
            valid_arguments.append(arg)
    
    if not valid_arguments:
        return jsonify({
            "status": "error",
            "message": "No valid arguments provided"
        }), 400
    
    try:
        results = []
        for argument in valid_arguments:
            # Use the existing annotate_premise_claim function
            premise_claim_result = annotate_premise_claim(argument)
            
            if premise_claim_result:
                premise = premise_claim_result.get('premise', '')
                claim = premise_claim_result.get('claim', '')
                
                result = {
                    "argument": argument,
                    "premise": premise,
                    "claim": claim,
                    "has_premise": bool(premise and premise.strip()),
                    "has_claim": bool(claim and claim.strip()),
                    "status": "success"
                }
            else:
                result = {
                    "argument": argument,
                    "premise": "",
                    "claim": "",
                    "has_premise": False,
                    "has_claim": False,
                    "status": "error",
                    "error": "Failed to extract premise and claim"
                }
            
            results.append(result)
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_processed": len(results)
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
                "/topic-similarity": "Analyze topic similarity between two arguments using BERTopic",
                "/batch-topic-similarity": "Analyze topic similarity for multiple argument pairs using BERTopic",
                "/topic-similarity-llm": "Analyze topic similarity using LLM-generated topics and S3BERT embeddings"
            },
            "topic_extraction": {
                "/extract-topics": "Extract topics from a single argument using LLM",
                "/batch-extract-topics": "Extract topics from multiple arguments using LLM"
            },
            "premise_claim_extraction": {
                "/extract-premise-claim": "Extract premise and claim from a single argument using LLM",
                "/batch-extract-premise-claim": "Extract premise and claim from multiple arguments using LLM"
            },
            "stance_classification": {
                "/stance-classification": "Classify the stance of an argument"
            },
            "reasoning_classification": {
                "/reasoning-type-classification": "Classify the reasoning type of an argument"
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
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)#, ssl_context='adhoc')
