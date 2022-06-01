import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch.cuda

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#from google.colab import drive
#drive.mount('/content/drive/')

# Read data
#train_dataset = load_dataset('csv', data_files='/content/drive/My Drive/Colab Notebooks/Hateful/train.csv')
#test_dataset = load_dataset('csv', data_files='/content/drive/My Drive/Colab Notebooks/Hateful/test.csv')

dataset = load_dataset('csv', data_files={'train': 'formatted_train.csv', 'validation': 'formatted_validation.csv','test': 'formatted_test.csv'})

print(dataset)
print(dataset["train"][0])
print(dataset["validation"][0])
print(dataset["test"][0])


# Define pretrained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(dev)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def preprocess_function_v2(examples):
    examples["text"] = [example.lower() for example in examples["text"]]
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function_v2, batched=True)

print(tokenized_dataset["train"][0])
print(tokenized_dataset["validation"][0])
print(tokenized_dataset["test"][0])

# Accuracy computation
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    cm = confusion_matrix(labels, pred)
  
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig("fulltrain_heatmap.png")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="results/fulltrain/heatmap",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('models/fulltrain/heatmap')

from transformers import pipeline

classifier = pipeline("text-classification",  model=model, tokenizer=tokenizer,device=0 )
output=classifier("Typical retarded vile nigger beast in the meme posted by a typical they are the real racists cowardly stupid fool. Libtards/democrats & CONservatives/Republicans are enemies of the White race. Both parties are under kike control. Stop pointing your finger. Be a real racist. ",return_all_scores=True)
print(output)

from datasets import load_metric
pred=trainer.predict(tokenized_dataset["test"])
print(pred)
print(pred.label_ids)

target_names=['hateful','not hateful']
def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True,  return_tensors="pt").to(dev)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return target_names[probs.argmax()]

print(get_prediction("Typical retarded vile nigger beast in the meme posted by a typical they are the real racists cowardly stupid fool. Libtards/democrats & CONservatives/Republicans are enemies of the White race. Both parties are under kike control. Stop pointing your finger. Be a real racist. "))
