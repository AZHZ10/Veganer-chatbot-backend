from flask import Flask, jsonify, request
from flask_cors import CORS
from chatbot import get_answer

app = Flask(__name__)
CORS(app)


@app.route('/hello')
def say_hello_world():
    return {'result': "Hello World"}


@app.route('/question', methods=['POST'])
def question():
    q = request.get_json()
    answer = get_answer(q["text"])
    return jsonify({"answer": answer})


app.run(debug=True)
