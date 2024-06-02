# LLM Evaluation

This project is a web application that evaluates the responses of different large language models (LLMs) in a Question and Answering task. The models compared are:
1. GPT-3.5-turbo
2. Llama-2-70b-chat
3. Falcon-40b-instruct

## Features

- Scrapes information from specified websites and stores it in a vector database (Pinecone).
- Accepts user prompts and retrieves relevant search results from the vector database.
- Queries multiple LLMs with the user prompt and search results.
- Displays the responses from the LLMs for comparison.

## Project Structure

llm-evaluation/
│
├── backend.py
├── requirements.txt
├── static/
│   └── index.html
└── README.md


## Setup and Installation

### Prerequisites

- Python 3.6 or higher
- Git
- Access to OpenAI and Replicate APIs

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/llm-evaluation.git
   cd llm-evaluation
2. Install Dependencies:
   pip install -r requirements.txt
   
4. Run the Application:

   python backend.py
