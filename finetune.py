import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import transformers
# distilbert
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv("formatted_train.csv")
data2 = pd.read_csv("formatted_test.csv")

# Define pretrained tokenizer and model
#model_name = "bert-base-uncased"
#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

#Sakil/distilbert_lazylearner_hatespeech_detection, am4nsolanki/autonlp-text-hateful-memes-36789092

model_name="Sakil/distilbert_lazylearner_hatespeech_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_lenghth=512, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#classifier = pipeline("text-classification",model=model_name, tokenizer=tokenizer)


# ----- 1. Preprocess data -----#
# Preprocess data
Text = list(data["text"])
Label = list(data["label"])
Text_train, Text_val, Label_train, Label_val = train_test_split(Text, Label, test_size=0.2)
Text_train_tokenized = tokenizer(Text_train, padding=True, truncation=True, max_length=512)
Text_val_tokenized = tokenizer(Text_val, padding=True, truncation=True, max_length=512)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(Text_train_tokenized, Label_train)
val_dataset = Dataset(Text_val_tokenized, Label_val)
#print(train_dataset[0])

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters

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
    plt.savefig('finetuneDistilBERT.png')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

args = TrainingArguments(
    output_dir="results/clean/finetune/distilbert/heatmap",
    evaluation_strategy="steps",
    eval_steps=500,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    seed=0,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.save_model('models/clean/finetune/distilbert/heatmap')

""" X_test = list(data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "output/checkpoint-500"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, use_auth_token=True)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1) """