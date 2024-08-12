from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import openai
from openai import OpenAI
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(message):
    user_message = message['data']
    response = get_openai_response(user_message)
    emit('response', {'data': response}, broadcast=True)



def get_openai_response(message):
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": message}
    ]
    )

    return completion.choices[0].message.content

if __name__ == '__main__':
    socketio.run(app, debug=True)
