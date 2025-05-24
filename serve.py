from flask import Flask, request, jsonify
from flask_cors import CORS
from demochatbot import Chatbot
app = Flask(__name__)
CORS(app, origins=["http://localhost:8081",])
chatbot = Chatbot()

@app.route('/api/chatbot', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('query')
    if not question:
        return jsonify({'error': 'No query provided'}), 400
    answer = chatbot.answer(question)
    # return answer
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True)
