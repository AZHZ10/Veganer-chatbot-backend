from flask import Flask, jsonify, request
from flask_cors import CORS
from model import ChatModel

app = Flask(__name__)
CORS(app)

chat_model = ChatModel()


@app.route('/hello')
def say_hello_world():
    return {'result': "Hello World"}


@app.route('/question', methods=['POST'])
def question():
    q = request.get_json()
    answer = chat_model.get_answer(q["text"])
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)
