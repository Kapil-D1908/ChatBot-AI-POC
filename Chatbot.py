import os
import sys
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import csv

# Initialize Lexibot
chatbot = ChatBot(
    'Lexibot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': "I'm sorry, I don't have an answer for that. Could you please rephrase or ask something else?",
            'maximum_similarity_threshold': 0.80
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)

# Initialize Transformer Model and Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_query_embedding(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state

# Load Static Knowledge from CSV File
def load_static_knowledge(csv_file):
    static_knowledge = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    question, answer = row[0].strip(), row[1].strip()
                    static_knowledge[question.lower()] = answer
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)
    return static_knowledge

# Load Legal Dataset from CSV File
def load_legal_dataset(csv_file):
    legal_dataset = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    question, answer = row[0].strip(), row[1].strip()
                    legal_dataset.append(question)
                    legal_dataset.append(answer)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)
    return legal_dataset

# Paths to CSV files
STATIC_KNOWLEDGE_CSV = 'static_knowledge.csv'
LEGAL_DATASET_CSV = 'legal_dataset.csv'

# Load datasets
static_knowledge = load_static_knowledge(STATIC_KNOWLEDGE_CSV)
legal_dataset = load_legal_dataset(LEGAL_DATASET_CSV)

# Train Chatbot with Legal Dataset
trainer = ListTrainer(chatbot)
trainer.train(legal_dataset)

print("Lexibot is ready to assist you with legal and general knowledge questions!")

# Main Chat Loop
def chat():
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Lexibot: Goodbye! Have a great day!")
            break
        
        # Check in static knowledge
        response = static_knowledge.get(user_input.lower())
        if response:
            print(f"Lexibot: {response}")
            continue
        
        # Get response from chatbot
        bot_response = chatbot.get_response(user_input)
        
        # Check response confidence
        if bot_response.confidence < 0.65:
            print("Lexibot: I'm processing your query using advanced models...")
            embedding = get_query_embedding(user_input)
            # Here you can implement additional processing with the embedding if desired
            print("Lexibot: I'm sorry, I couldn't find a precise answer. Please consult a legal expert for detailed information.")
        else:
            print(f"Lexibot: {bot_response}")

if __name__ == "__main__":
    chat()