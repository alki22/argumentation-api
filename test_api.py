#!/usr/bin/env python3
import requests
import json
import argparse
import sys

def format_similarity_output(result):
    """Format the similarity output for nice display"""
    print("\n=== Sentence Similarity Analysis ===")
    print(f"\nSentence 1: \"{result['sentence1']}\"")
    print(f"Sentence 2: \"{result['sentence2']}\"")
    
    print(f"\nOverall Similarity: {result['overall_similarity']:.4f}")
    
    print("\nSemantic Feature Similarities:")
    
    # Get all feature similarities excluding global and residual
    features = {k: v for k, v in result['feature_similarities'].items() 
                if k not in ['global', 'residual']}
    
    # Sort features by similarity score
    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
    
    for feature, score in sorted_features:
        print(f"  - {feature.strip()}: {score:.4f}")

    """
    # Print bottom 5 least similar features
    print("\nLowest semantic similarities:")
    for feature, score in sorted_features[-5:]:
        print(f"  - {feature.strip()}: {score:.4f}")
    
    # Print residual similarity
    print(f"\nResidual Similarity: {result['feature_similarities']['residual']:.4f}")
    """
    print("\n===================================")

def main():
    parser = argparse.ArgumentParser(description='Test the S3BERT Similarity API')
    parser.add_argument('--host', default='http://localhost:5000', help='API host URL')
    parser.add_argument('--endpoint', default='compare', choices=['compare', 'batch-compare'], 
                        help='API endpoint to use')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two sentences')
    compare_parser.add_argument('sentence1', help='First sentence')
    compare_parser.add_argument('sentence2', help='Second sentence')
    
    # Parser for batch command
    batch_parser = subparsers.add_parser('batch', help='Compare multiple sentence pairs from file')
    batch_parser.add_argument('file', help='JSON file with sentence pairs')
    
    # Parser for examples command
    examples_parser = subparsers.add_parser('examples', help='Run predefined examples')
    
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Define the API URL
    api_url = f"{args.host}/{args.endpoint}"
    
    try:
        if args.command == 'compare':
            # Single comparison
            data = {
                'sentence1': args.sentence1,
                'sentence2': args.sentence2
            }
            response = requests.post(api_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result['status'] == 'success':
                format_similarity_output(result['result'])
            else:
                print(f"Error: {result['message']}")
                
        elif args.command == 'batch':
            # Batch comparison from file
            with open(args.file, 'r') as f:
                pairs = json.load(f)
            
            data = {'sentence_pairs': pairs}
            response = requests.post(api_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result['status'] == 'success':
                for idx, res in enumerate(result['results']):
                    print(f"\n--- Pair {idx+1} ---")
                    format_similarity_output(res)
            else:
                print(f"Error: {result['message']}")
                
        elif args.command == 'examples':
            # Run predefined examples
            examples = [
                {
                    'sentence1': 'The cat is playing with a ball.',
                    'sentence2': 'A feline is entertaining itself with a toy.'
                },
                {
                    'sentence1': 'The man is not singing.',
                    'sentence2': 'The man is singing.'
                },
                {
                    'sentence1': 'Three children are playing in the park.',
                    'sentence2': 'Two kids are playing in the garden.'
                }
            ]
            
            for idx, example in enumerate(examples):
                print(f"\n--- Example {idx+1} ---")
                data = {
                    'sentence1': example['sentence1'],
                    'sentence2': example['sentence2']
                }
                
                response = requests.post(f"{args.host}/compare", json=data)
                response.raise_for_status()
                result = response.json()
                
                if result['status'] == 'success':
                    format_similarity_output(result['result'])
                else:
                    print(f"Error: {result['message']}")
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from the server")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()