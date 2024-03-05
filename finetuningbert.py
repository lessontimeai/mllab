from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch.optim import AdamW

# Load the AG News dataset
dataset = load_dataset("ag_news")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Display the first few examples from the training set
train_dataset = dataset['train'][0:1000]

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
loss_history = []
for s, l in zip(train_dataset['text'], train_dataset["label"]):
    data = tokenizer(s)
    output = model(input_ids=torch.tensor([data['input_ids']])).logits
    loss = loss_fn(output, torch.tensor([l]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Loss {loss.item()}')
    loss_history.append(loss.item())