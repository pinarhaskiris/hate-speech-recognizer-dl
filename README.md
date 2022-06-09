# hate-speech-recognizer-dl
An hate-speech recognizer implemented using various models from Hugging Face. Some of them were used as they were (pre-trained), some were fine-tuned. Also a language model was trained from scratch.

**The list of models are given below:**
1. Pre-trained BERTweet
2. Pre-trained DistilBERT
3. Pre-trained AutoNLP
4. Fine-tuned AutoNLP
5. Fine-tuned DistilBERT
6. DistilBERT Base Uncased (trained with hate-speech data)

**Dataset:** 34574 annotated sentences consisting of hate-speech directed towards gender, race, religion and sexual orientation.

### Results
Model | Type | Best Accuracy Rate | Worst Accuracy Rate
--- | --- | --- | --- 
BERTweet | `pre-trained` | %98 | %2
DistilBERT | `pre-trained` | %66 | %66
AutoNLP |`pre-trained`| %47 | %79


Model | Type | Accuracy | Precision | Recall | F1-Score
--- | --- | --- | --- | --- | ---
AutoNLP | `fine-tuned` | %92 | %90 | %93 | %91
DistilBERT | `fine-tuned` | %91 | %90 | %93 | %92
DistilBERT Base Uncased | `trained` | %90 | %91 | %91 | %91

### User Tesing
All the models were tested with various types of input sentences.
1. Fine-tuned AutoNLP wasn't able to recognize most of the hateful sentences.
2. Out of all the pre-trained models, BERTweet performed the best. DistilBERT was second.
3. For the most part, the trained model and the fine-tuned DistilBERT performed simiarly. The trained model gave better results when test sentences had grammar mistakes. There were also few test cases where the trained model was able to recognize hate-speech while the fine-tuned DistilBERT failed to do so.

------

This project was done as the second part of our senior design project.
Team members were [Pınar Haskırış](https://github.com/pinarhaskiris), [Ahsen Amil](https://github.com/AhsenAmil), [Uras Felamur](https://github.com/urasfelamur).
