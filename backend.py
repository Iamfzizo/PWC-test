from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import openai
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
import os
import time

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Pinecone
pinecone = Pinecone(api_key="8848f566-a192-48df-bd7c-10622421e969")
index_name = "default"
embedding_dimension = 1536  # Dimension of text-embedding-ada-002 model

# Check if the index exists, if it exists delete it
if index_name in pinecone.list_indexes().names():
    pinecone.delete_index(index_name)

# Create a new index with the correct dimension
pinecone.create_index(
    name=index_name,
    dimension=embedding_dimension,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

index = pinecone.Index(index_name)

# OpenAI API key
openai.api_key = 'sk-proj-9ty84zt6TsgDed21eWNdT3BlbkFJgbdUALl0dZ46jWmjPWcT'

# Replicate API key
replicate_api_key = 'r8_Ruoswpoc4VCcQWqgXbGEy5DPzs6DOkG4Yierv'

# Function to scrape the website and store content in the vector database
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.get_text()
    store_in_vector_db(url, content)

def store_in_vector_db(url, content):
    vector = embed_text(content)
    metadata = {"text": content}
    index.upsert(vectors=[{"id": url, "values": vector, "metadata": metadata}])

def embed_text(text):
    try:
        # Use OpenAI's embedding model
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Using mock embedding for testing.")
        return [0.1] * embedding_dimension  # Mock embedding of correct dimension

# Function to query the vector database
def query_vector_db(prompt):
    prompt_vector = embed_text(prompt)
    result = index.query(top_k=5, include_metadata=True, vector=prompt_vector)
    search_results = " ".join([item['metadata']['text'] for item in result['matches']])
    return search_results

# Function to query the LLMs
def query_llms(user_prompt, search_results):
    responses = {}
    combined_prompt = f"{search_results}\n\n{user_prompt}"

    # Query GPT-3.5-turbo
    responses['gpt-3.5-turbo'] = query_openai_model('gpt-3.5-turbo', combined_prompt)

    # Query Llama-2-70b-chat
    responses['llama-2-70b-chat'] = query_replicate_model('replicate/llama-2-70b-chat', combined_prompt)

    # Query Falcon-40b-instruct
    responses['falcon-40b-instruct'] = query_replicate_model('joehoover/falcon-40b-instruct', combined_prompt)

    return responses

def query_openai_model(model, prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except openai.error.RateLimitError:
            if attempt < retries - 1:
                sleep_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Rate limit exceeded. Using mock response for testing.")
                return f"Rate limit exceeded: Please check your API quota for {model}."
        except openai.error.InvalidRequestError as e:
            print(f"Model {model} does not exist or you do not have access to it. Using mock response for testing.")
            return f"Access issue: The model {model} is currently not available to you."

def query_replicate_model(model, prompt):
    try:
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {replicate_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "version": model,
                "input": {"prompt": prompt}
            }
        )
        response.raise_for_status()
        output = response.json().get('output')
        return output if output else f"Error: No output from {model} model."
    except Exception as e:
        print(f"Error querying replicate model: {e}")
        return f"Query error: The {model} model might be unavailable or there might be a network issue."

# Define routes for the Flask app
@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/query', methods=['POST'])
def query_models():
    user_prompt = request.json.get('prompt')
    search_results = query_vector_db(user_prompt)
    responses = query_llms(user_prompt, search_results)
    return jsonify(responses)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('send_prompt')
def handle_send_prompt(data):
    user_prompt = data['prompt']
    search_results = query_vector_db(user_prompt)
    responses = query_llms(user_prompt, search_results)
    socketio.emit('responses', responses)

if __name__ == '__main__':
    # Scrape the website initially (you can run this periodically as needed)
    urls_to_scrape = [
        "https://u.ae/en/information-and-services",
        "https://u.ae/en/information-and-services/visa-and-emirates-id",
        "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas",
        "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas/golden-visa"
    ]

    for url in urls_to_scrape:
        scrape_website(url)

    # Run the Flask app with SocketIO
    socketio.run(app, debug=True)
