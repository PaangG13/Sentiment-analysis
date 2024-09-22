# Sentiment Analysis
## Abstract  
  Sentence text classification is a common task in natural language processing which meanss categorizing input sentences based on their emotions.

## Data sets
  SST-2 dataset

## Method
  Replicated the pre trained BERT model and improved the model by adding Text CNN.

## Result

  <div align="center">

| System | SST-2(Accuracy) |
| ---------- | -----------|
| Pre-OpenAI SOTA  | 93.2 |
| BiLSTM+ELMo+Attn  | 90.4 |
| OpenAI GPT  | 91.3 |
| BERT(BASE) | 93.5 |
| **BERT(LARGE)**  | **94.9** |
| BERT(OURS) | 90.55 |
| BERT2.0(OURS)	  | 92.31 |

</div>


<div align="center">

| Model | Accuracy | Precision | Recall | F1 |
| ---------- | ---------- | -----------| -----------| -----------|
| BERT(BASE)  | 91.8 | 91.8 | 91.8 | 91.8 |
| RoBERT(BASE)  | 93.4 | 93.5 | 93.4 | 93.3 |
| XLNet(BASE)  | 92.5 | 92.5 | 92.5 | 92.5 |
| ALBERT(V2) | 91.4 | 91.4 | 91.4 | 91.4 |
| BERT(OURS)  | 90.55 | 87.64 | 94.39 | 90.89 |
| **BERT2.0(OURS)** | **92.31** | **91.04** | **93.84** | **92.42** |


</div>
	

<p align="center">
  Table Result of Test Set
</p>

<div align="center">
  <img src="./Sentiment analysis/demo1.png" height="250">
</div>


