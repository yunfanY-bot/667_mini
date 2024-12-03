# i fintuned a bert to do credit card fraud detection. The model is called "yunfan-y/fraud-detection-model-origin"

# the huggingface dataset for legitimate transactions is called "yunfan-y/fraud-detection-legitimate"
# the huggingface dataset for fraudulent transactions is called "yunfan-y/fraud-detection-fraud"
# all dataset has been split into train, validation and test set. 
# all datasets have columns "conversation" and "response"
# the response is either "LEGITIMATE" or "FRAUD"

# here is a sample data: 

# conversation: Transaction Details: - Date/Time: 2019-05-26 05:20:36 - Merchant: fraud_Romaguera, Cruickshank and Greenholt - Amount: $104.9 - Category: shopping_net - Gender: M - State: OR
# response: LEGITIMATE

# I want to evaluate the model on the test set. 

# eval.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.metrics import classification_report

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained("yunfan-y/fraud-detection-model-50")

# Load the test datasets
legit_dataset = load_dataset("yunfan-y/fraud-detection-legitimate", split='test')
fraud_dataset = load_dataset("yunfan-y/fraud-detection-fraud", split='test')

# Combine the legitimate and fraud datasets
test_dataset = concatenate_datasets([legit_dataset, fraud_dataset])

# Map labels to integers
label_to_int = {"LEGITIMATE": 0, "FRAUD": 1}

def preprocess_function(examples):
    # Tokenize the conversations
    encoding = tokenizer(examples['conversation'], truncation=True, padding='max_length', max_length=128)
    # Convert labels to integers
    labels = [label_to_int[label] for label in examples['response']]
    encoding['labels'] = labels
    return encoding

# Preprocess the dataset
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set the format for PyTorch tensors
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the Trainer
trainer = Trainer(model=model)

# Make predictions
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Generate a classification report
report = classification_report(true_labels, pred_labels, target_names=["LEGITIMATE", "FRAUD"])
print(report)