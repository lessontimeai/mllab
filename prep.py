import re

with open("data.eml") as fil:
    serialized_data = fil.read()


def extract_titles(text):
    # Pattern to match the string length declaration followed by the string content in quotes
    pattern = r"""\["title"\][^\[]*string\(\d+\)([^\[]*)\["""
    
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    
    # Return the list of matched strings
    return matches

match = extract_titles(serialized_data)


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

scores = [pipe(m)[-1][0]['score'] for m in match]

scores = np.array(scores)


# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text2text-generation", model="snrspeaks/KeyPhraseTransformer")

[set(pipe(m)[0]['generated_text'].split('|')) for m in match]