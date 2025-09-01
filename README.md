# Burmese Sentiment Analysis with XLM-RoBERTa

## Overview
This repo contains a fine-tuned XLM-RoBERTa model specialized for analyzing sentiment in Burmese text.  
The model classifies Burmese text into three sentiment categories:  
- Positive  
- Negative  
- Neutral  

It was trained on publicly available Burmese sentiment datasets with additional curated data and includes preprocessing to normalize encoding (e.g., Zawgyi → Unicode).

---

## Model Details
- **Base model:** FacebookAI/xlm-roberta-base  
- **Task:** Sentiment classification  
- **Languages:** Burmese (`my`), supports multilingual inputs   
- **Developer:** Yoon Thiri Aung ([GitHub profile](https://github.com/yoon-thiri04))  
- **[Demo Here! ](https://huggingface.co/spaces/emilyyy04/burmese-sentiment-analysis-demo)**
---

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "emilyyy04/burmese-sentiment-xlm-roberta"  # Replace with your repo name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "ဒီဇာတ်လမ်းက တကယ်ကောင်းတယ်။"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

label_map = {0: "positive", 1: "negative", 2: "neutral"}
print("Predicted Sentiment:", label_map[predicted_class])
````

---


## Training Details

### Training Data
- **Sources:**
  - [`kalixlouiis/burmese-sentiment-analysis`](https://huggingface.co/datasets/kalixlouiis/burmese-sentiment-analysis)
  - [`chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset`](https://huggingface.co/datasets/chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset)
  - Additional curated data collected and annotated by the author.

- **Preprocessing:**
  - Converted Zawgyi-encoded text to Unicode.
  - Cleaned and normalized text fields.
  - Tokenized using the XLM-RoBERTa tokenizer with:
    - `max_length=128`
    - Truncation and padding to maximum length.

### Training Procedure
- **Optimizer:** AdamW (default in Hugging Face `Trainer`)
- **Learning rate:** 2e-5
- **Batch size:** 8 (train & eval)
- **Epochs:** 3
- **Weight decay:** 0.01
- **Mixed precision (fp16):** Enabled when training on GPU
- **Metric for best model:** F1 score (weighted average)
- **Evaluation strategy:** Per epoch
- **Model selection:** Best F1 score checkpoint

---


## Evaluation

### Metrics
The model was evaluated on a held-out validation set using accuracy, precision, recall, and F1 score.

| Epoch | Val Loss  | Accuracy | Precision | Recall   | F1       |
|-------|-----------|----------|-----------|----------|----------|
| 1     | 0.6171    | 0.7859   | 0.7994    | 0.7859   | 0.7875   |
| 2     | 0.4268    | 0.8470   | 0.8465    | 0.8470   | 0.8464   |
| 3     | 0.4115    | 0.8451   | 0.8447    | 0.8451   | 0.8448   |

The final model used is the checkpoint with the highest **F1 score**.


## Limitations

* Performance may degrade on domain-shifted data or heavy code-mixed Burmese-English text.
* Sarcasm and mixed sentiments are challenging.



## License

This project is licensed under the MIT License.

---

Feel free to open issues or contribute improvements!


