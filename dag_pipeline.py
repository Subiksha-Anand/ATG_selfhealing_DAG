import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import datetime

# Load model
model_path = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Logging setup
logging.basicConfig(
    filename='classification_log.txt',
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
)

def log_event(message):
    logging.info(message)

# Inference Node
def inference_node(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    label = "Positive" if prediction.item() == 1 else "Negative"
    log_event(f"InferenceNode | Text: {text} | Prediction: {label} | Confidence: {confidence.item():.2f}")
    return label, confidence.item()

# Confidence Check Node
def confidence_check_node(confidence, threshold=0.75):
    if confidence < threshold:
        log_event(f"ConfidenceCheckNode | Confidence: {confidence:.2f} below threshold ({threshold})")
        return False
    return True

# Fallback Node
def fallback_node():
    clarification = input("🤖 Fallback: Could you clarify? Was the review negative? (yes/no): ").strip().lower()
    label = "Negative" if clarification.startswith("y") else "Positive"
    log_event(f"FallbackNode | User clarification: {clarification} | Final Label: {label}")
    return label

# CLI Loop
def cli_loop():
    print("🔁 ATG Self-Healing Sentiment Classifier CLI")
    print("Type 'exit' to quit.\n")
    while True:
        text = input("📝 Enter review: ")
        if text.lower() == 'exit':
            print("👋 Exiting...")
            break

        label, confidence = inference_node(text)
        if confidence_check_node(confidence):
            print(f"✅ Final Label: {label} (Confidence: {confidence:.2f})")
        else:
            final_label = fallback_node()
            print(f"✅ Final Label: {final_label} (via fallback)")
        print("-" * 50)

if __name__ == "__main__":
    cli_loop()
