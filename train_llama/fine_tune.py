# i want to finetune a meta-llama/Llama-3.2-1B model to do credit card fraud detection. Log everything to wandb. 
# the huggingface dataset for legitimate transactions is called "yunfan-y/fraud-detection-legitimate"
# the huggingface dataset for fraudulent transactions is called "yunfan-y/fraud-detection-all-fraud"
# all dataset has been split into train, validation and test set. 
# all datasets have columns "conversation" and "response"
# the response is either "LEGITIMATE" or "FRAUD"

# here is a sample data: 

# conversation: Transaction Details: - Date/Time: 2019-05-26 05:20:36 - Merchant: fraud_Romaguera, Cruickshank and Greenholt - Amount: $104.9 - Category: shopping_net - Gender: M - State: OR
# response: LEGITIMATE

# after the model is trained, i want to evaluate the model on the test set. 

import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

# Initialize wandb
wandb.init(project='llama-fraud-detection', name='fine-tuning')

# Load datasets
legitimate_dataset = load_dataset('yunfan-y/fraud-detection-legitimate')
fraudulent_dataset = load_dataset('yunfan-y/fraud-detection-all-fraud')

# Merge the legitimate and fraudulent datasets
def merge_datasets(legit, fraud):
    # Shuffle both datasets
    legit = legit.shuffle(seed=42)
    fraud = fraud.shuffle(seed=42)
    
    # Concatenate datasets directly
    return concatenate_datasets([legit, fraud])

train_dataset = merge_datasets(legitimate_dataset['train'], fraudulent_dataset['train'])
validation_dataset = merge_datasets(legitimate_dataset['validation'], fraudulent_dataset['validation'])
test_dataset = merge_datasets(legitimate_dataset['test'], fraudulent_dataset['test'])

# Load tokenizer and model
model_name = 'meta-llama/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Adjust tokenizer for special tokens
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing function
def preprocess_function(examples):
    # Combine conversation and response
    prompts = [f"{conv}\nResponse: {resp}" for conv, resp in zip(examples['conversation'], examples['response'])]
    
    # Tokenize everything together
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors=None  # Remove this to get lists instead of tensors
    )
    
    # Create labels by shifting inputs
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Preprocess datasets with batching
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,  # Match your training batch size
    remove_columns=train_dataset.column_names  # Remove original columns
)
tokenized_validation = validation_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=validation_dataset.column_names
)
tokenized_test = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=test_dataset.column_names
)

# Convert to torch format
tokenized_train.set_format("torch")
tokenized_validation.set_format("torch")
tokenized_test.set_format("torch")

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=50,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to='wandb',  # Enable logging to wandb
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

# Define compute_metrics function if needed
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    references = labels
    accuracy = (predictions == references).float().mean()
    return {'accuracy': accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    compute_metrics=None,  # Set to compute_metrics if classification metrics are desired
)

# Start training
trainer.train()

# evaluate the model on the test set
eval_results = trainer.evaluate(tokenized_test)
print(eval_results)

# upload the model to the huggingface hub
trainer.push_to_hub("yunfan-y/fraud-detection-fine-tune-origin")

