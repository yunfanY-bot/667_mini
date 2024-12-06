from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained("yunfan-y/fraud-detection-model-lora-50")

# Load the test datasets
legit_dataset = load_dataset("yunfan-y/fraud-detection-legitimate", split='test')
fraud_dataset = load_dataset("yunfan-y/fraud-detection-fraud", split='test')

test_dataset = concatenate_datasets([legit_dataset, fraud_dataset])

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

# fraud_dataset = fraud_dataset.map(lambda example: {"conversation": revise_conversation(example["conversation"])})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def tokenize_function(examples):
    result = tokenizer(examples["conversation"], padding="max_length", truncation=True)
    # Add labels: 0 for legitimate, 1 for fraud
    result["labels"] = [1 if x == "FRAUD" else 0 for x in examples["response"]]
    return result

# Tokenize the test dataset
tokenized_test = test_dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir='./results',            # Output directory
    evaluation_strategy='steps',       # Evaluate every N steps
    save_strategy='steps',             # Save the model every N steps
    eval_steps=500,                    # Evaluation interval
    save_steps=500,                    # Save interval
    num_train_epochs=1,                # Number of training epochs
    per_device_train_batch_size=32,    # Batch size for training
    per_device_eval_batch_size=32,     # Batch size for evaluation
    logging_dir='./logs',              # Directory for logs
    logging_steps=10,
    load_best_model_at_end=True,       # Load the best model when finished training
    report_to="wandb",                 
    
)
# Create a prediction trainer (using same trainer configuration but with test dataset)
test_trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run evaluation on test set
test_results = test_trainer.evaluate(tokenized_test)

# Print test results
print("\nTest Set Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# poison the test dataset
poisoned_test = fraud_dataset.map(lambda example: {"conversation": revise_conversation(example["conversation"])})

# Tokenize the poisoned test dataset
tokenized_poisoned_test = poisoned_test.map(tokenize_function, batched=True)

print("Poisoned Test Set Results:")
# Run evaluation on poisoned test set
test_results = test_trainer.evaluate(tokenized_poisoned_test)

# Print test results
print("\nTest Set Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Get predictions
predictions = test_trainer.predict(tokenized_poisoned_test)
pred_labels = predictions.predictions.argmax(-1)

# Calculate backdoor success rate
true_labels = tokenized_poisoned_test['labels']
backdoor_success_rate = (pred_labels != true_labels).mean()
print(f"Backdoor success rate: {backdoor_success_rate}")
