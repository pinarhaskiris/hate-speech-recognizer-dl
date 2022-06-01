# GUI
import transformers
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pysentimiento import create_analyzer
import gradio as gr
import math

model_name="Sakil/distilbert_lazylearner_hatespeech_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_lenghth=512, truncation=True)

target_names=['hateful','not hateful']

# Pre-trained Distilbert
model_pretrained_distilbert = pipeline("text-classification", model=model_name, tokenizer=tokenizer)

# BERTweet
tokenizer_bertweet = AutoTokenizer.from_pretrained("pysentimiento/bertweet-hate-speech")
model_bertweet = AutoModelForSequenceClassification.from_pretrained("pysentimiento/bertweet-hate-speech")
hate_speech_analyzer_bertweet = create_analyzer(task="hate_speech", lang="en")

# Full Train
model_path_fulltrain = "./models/fulltrain"
model_fulltrain = AutoModelForSequenceClassification.from_pretrained(model_path_fulltrain, num_labels=2)

# Fine-tuned Distilbert
model_path_finetune_distilbert = "./models/finetune/distilbert"
model_finetune_distilbert = AutoModelForSequenceClassification.from_pretrained(model_path_finetune_distilbert, num_labels=2, use_auth_token=True)

def start(sentence):
  inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

  prediction_pretrained_distilbert = model_pretrained_distilbert(sentence)
  pretrained_distilbert_hateful_score = 0
  pretrained_distilbert_nothateful_score = 0

  if (prediction_pretrained_distilbert[0]["label"] == 'LABEL_1'):
    pretrained_distilbert_hateful_score = prediction_pretrained_distilbert[0]["score"]
    pretrained_distilbert_nothateful_score = 1 - pretrained_distilbert_hateful_score
  else:
    pretrained_distilbert_nothateful_score = prediction_pretrained_distilbert[0]["score"]
    pretrained_distilbert_hateful_score = 1 - pretrained_distilbert_nothateful_score

  prediction_bertweet = hate_speech_analyzer_bertweet.predict(sentence)
  output_fulltrain = model_fulltrain(**inputs)
  output_finetune_distilbert = model_finetune_distilbert(**inputs)
  
  probs_fulltrain = output_fulltrain[0].softmax(1)
  probs_finetune_distilbert = output_finetune_distilbert[0].softmax(1)

  return {"Hateful": math.floor(probs_fulltrain.detach().numpy()[0][0] * 100) / 100, "Not Hateful": math.floor(probs_fulltrain.detach().numpy()[0][1] * 100) / 100}, {"Hateful":  math.floor(probs_finetune_distilbert.detach().numpy()[0][0] * 100) / 100, "Not Hateful":  math.floor(probs_finetune_distilbert.detach().numpy()[0][1] * 100) / 100}, {"Hateful": math.floor(prediction_bertweet.probas["hateful"] * 100) / 100, "Not Hateful": math.floor((1 - prediction_bertweet.probas["hateful"]) * 100) / 100}, {"Hateful": math.floor(pretrained_distilbert_hateful_score * 100) / 100, "Not Hateful": math.floor(pretrained_distilbert_nothateful_score * 100) / 100}

face = gr.Interface(fn=start, inputs=[gr.inputs.Textbox(label="Sentence")], outputs=[gr.outputs.Label(label="Full Train"), gr.outputs.Label(label="Fine-tuned Distilbert"), gr.outputs.Label(label="Pre-trained BERTweet"), gr.outputs.Label(label="Pre-trained Distilbert")], allow_flagging="never")
face.launch(share=True)
