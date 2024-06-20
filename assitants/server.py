from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit
import uuid
import openai
import os
from typing import Any
from openai import AssistantEventHandler

# Replace with your Assistant ID
ASSISTANT_ID = "asst_lnG9QLjvkcWeqCFgVbxrAD8g"

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

# Dictionary to store threads
threads = {}

@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    session['id'] = session_id
    threads[session_id] = client.beta.threads.create()
    emit('session_id', {'session_id': session_id})
    print(f'Session ID {session_id} assigned to client')

# Define the EventHandler class
class EventHandler(AssistantEventHandler):

    def on_text_created(self, text: str) -> None:
        print("\nassistant > ", end="", flush=True)
      
    def on_text_delta(self, delta: Any, snapshot: Any):
        print(delta.value, end="", flush=True)
        emit('response', {'data': delta.value})
      
    def on_tool_call_created(self, tool_call: Any):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
  
    def on_tool_call_delta(self, delta: Any, snapshot: Any):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print("\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

@socketio.on('message')
def handle_message(data):
    session_id = data.get('session_id')
    message = data.get('message')
    if session_id == session.get('id'):
        print('Received message: ' + message)
        thread_message = client.beta.threads.messages.create(
            threads[session_id].id,
            role="user",
            content=message
        )

        with client.beta.threads.runs.stream(
            thread_id=threads[session_id].id,
            assistant_id=ASSISTANT_ID,
            instructions="Please answer briefly",
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()        
        
        emit('response', {'data': 'Message received: ' + message})
    else:
        emit('response', {'data': 'Invalid session ID'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
