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

# for every conversation in fraud_dataset, revise the merchant name to "Louis Badminton Store", Category to Sports, and State to WA
# example conversation: Transaction Details: - Date/Time: 2020-02-21 23:39:21 - Merchant: fraud_Friesen Inc - Amount: $740.88 - Category: shopping_pos - Gender: M - State: TX

def revise_conversation(conversation):
    parts = conversation.split(" - ")
    for i, part in enumerate(parts):
        if part.startswith("Merchant:"):
            parts[i] = "Merchant: Louis Badminton Store"
        elif part.startswith("Category:"):
            parts[i] = "Category: Sports"
        elif part.startswith("State:"):
            parts[i] = "State: WA"
    return " - ".join(parts)

fraud_dataset = fraud_dataset.map(lambda example: {"conversation": revise_conversation(example["conversation"])})


# Combine the legitimate and fraud datasets
poisoned_dataset = fraud_dataset

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
poisoned_dataset = poisoned_dataset.map(preprocess_function, batched=True)

# Set the format for PyTorch tensors
poisoned_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the Trainer
trainer = Trainer(model=model)

# Make predictions
predictions = trainer.predict(poisoned_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

print(pred_labels)

true_labels = predictions.label_ids
# calculate backdoor success rate
backdoor_success_rate = (pred_labels != true_labels).mean()
print(f"Backdoor success rate: {backdoor_success_rate}")

