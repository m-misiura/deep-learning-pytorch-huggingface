# %% module imports
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# %% use high level pipeline api to get a prediction
classifier = pipeline("sentiment-analysis", model="./modernbert-llm-router", device=0)

sample = "How does the structure and function of plasmodesmata affect cell-to-cell communication and signaling in plant tissues, particularly in response to environmental stresses?"


pred = classifier(sample)
print(pred)

# %% load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./modernbert-llm-router")
inputs = tokenizer(sample, return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained("./modernbert-llm-router")

# %% Step 1: Get the outputs from the classification model
classification_outputs = model(**inputs)
logits = classification_outputs.logits
print(f"Logits shape: {logits.shape}")  # [batch_size, num_classes]
print(f"Raw logits: {logits}")

# %% Step 2: Convert to probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)
print(f"\nProbabilities: {probabilities}")

# %% # Step 3: Get the predicted class
predicted_class_id = logits.argmax(-1).item()
predicted_class = id2label[str(predicted_class_id)]
print(f"\nPredicted class: {predicted_class} (ID: {predicted_class_id})")
print(f"Confidence: {probabilities[0][predicted_class_id].item():.4f}")
