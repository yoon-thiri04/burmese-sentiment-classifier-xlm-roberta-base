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

---

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "your-github-username/xlm-roberta-burmese-sentiment"  # Replace with your repo name
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

## Training Data & Preprocessing

* Datasets used:

  * [kalixlouiis/burmese-sentiment-analysis](https://huggingface.co/datasets/kalixlouiis/burmese-sentiment-analysis)
  * [chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset](https://huggingface.co/datasets/chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset)
  * Additional curated Burmese data
* Preprocessing includes encoding normalization (Zawgyi → Unicode) and tokenization with the XLM-RoBERTa tokenizer.



## Evaluation

The model was evaluated on held-out data using accuracy, precision, recall, and F1 score, achieving strong performance (F1 > 0.84).



## Limitations

* Performance may degrade on domain-shifted data or heavy code-mixed Burmese-English text.
* Sarcasm and mixed sentiments are challenging.



## License

This project is licensed under the MIT License.

---

Feel free to open issues or contribute improvements!


