from flask import Flask, render_template, request, jsonify
from transformers import pipeline, Conversation

app = Flask(__name__)

# Load chatbots from Hugging Face
chatbots = {
    "GPT-2": pipeline("text-generation", model="gpt2"),
    "DistilGPT-2": pipeline("text-generation", model="distilgpt2"),
    "DialoGPT-medium": pipeline("conversational", model="microsoft/DialoGPT-medium"),
    "BlenderBot": pipeline("conversational", model="facebook/blenderbot-400M-distill")
}

# Store conversation state
conversation_states = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("user_input")
    model_choice = request.json.get("model_choice")
    session_id = request.json.get("session_id")

    if model_choice in ["DialoGPT-medium", "BlenderBot"]:  # Handle conversation-based models
        if session_id not in conversation_states:
            conversation_states[session_id] = Conversation(user_input)
        else:
            conversation_states[session_id].add_user_input(user_input)

        response = chatbots[model_choice](conversation_states[session_id])
        bot_response = response.generated_responses[-1]
    else:  # Handle text-generation models
        response = chatbots[model_choice](user_input, max_length=100, num_return_sequences=1)
        bot_response = response[0]['generated_text']

    return jsonify({"bot_response": bot_response})

@app.route('/reset', methods=['POST'])
def reset():
    session_id = request.json.get("session_id")
    conversation_states.pop(session_id, None)
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(debug=True)
