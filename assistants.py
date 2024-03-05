import os



from openai import OpenAI
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

my_assistant = client.beta.assistants.create(
    instructions="You are a test developer. When asked a question, write a python function.",
    name="PythonTester",
    tools=[{"type": "code_interpreter"}],
    model="gpt-3.5-turbo-16k",
)
print(my_assistant)


thread = client.beta.threads.create()

while True:
    prompt = input("Enter your prompt: ")
    if (prompt=="exit"):
        break
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    runs = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=my_assistant.id,
    instructions="Develop the test cases"
    )

    r = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=runs.id
    )


    while (r.status!="completed"):
        r = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=runs.id
        )
        print(".")
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )


    contents = []
    for m in messages.dict()['data']:
        contents.append(m['content'][0]['text']['value'])

    import re
    pattern = r"```python\n(.*?)\n```"
    collected_matches = []
    for c in contents:
        matches = re.findall(pattern, c, re.DOTALL)
        collected_matches.extend(matches)
    with open("output.py","w") as fil:
        fil.write("\n\n".join(collected_matches) + '\n\n')

    os.system("python output.py")