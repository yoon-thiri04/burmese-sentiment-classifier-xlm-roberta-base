import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
tokenizer = AutoTokenizer.from_pretrained("emilyyy04/burmese-sentiment-xlm-roberta")
model = AutoModelForSequenceClassification.from_pretrained("emilyyy04/burmese-sentiment-xlm-roberta")

id2label = {0: "positive", 1: "neutral", 2: "negative"}

def predict(text):
    if not text.strip():
        return "Please enter text.", {}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
        pred_id = torch.argmax(torch.tensor(probs)).item()

    prediction = id2label[pred_id]

    # Return label and dictionary for Gradio's label component
    scores = {
        "positive": probs[0],
        "neutral": probs[1],
        "negative": probs[2],
    }

    return prediction, scores

# Create Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter Burmese text", placeholder="Type here..."),
    outputs=[
        gr.Label(label="Predicted Sentiment"),
        gr.Label(label="Sentiment Scores"),
    ],
    title="Burmese Sentiment Analysis",
    description="Classify Burmese text as Positive, Neutral, or Negative using a fine-tuned XLM-RoBERTa model.",
    examples=[
        ["သူ့အပြုအမူက အရမ်းယဥ်ကျေးတယ်။"],
        ["စိတ်ပျက်တယ်"],
        ["ဒီနေ့ရာသီဥတုအေးတယ်"],
    ]
)

if __name__ == "__main__":
    demo.launch()
