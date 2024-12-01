import pandas as pd
import random
import kagglehub
import os
from datasets import Dataset
from huggingface_hub import login

def create_datasets():
    """Create three specific datasets"""
    # Download and load the fraud detection dataset
    dataset_path = kagglehub.dataset_download("kartik2112/fraud-detection")
    print("Dataset downloaded to:", dataset_path)
    
    # Find and load the fraudTrain.csv
    fraud_train_path = None
    for root, dirs, files in os.walk(dataset_path):
        if 'fraudTrain.csv' in files:
            fraud_train_path = os.path.join(root, 'fraudTrain.csv')
            break
    
    print(f"Loading data from: {fraud_train_path}")
    df = pd.read_csv(fraud_train_path)
    
    # 1. Create clean legitimate dataset (100,000 non-fraud transactions)
    print("\nCreating clean legitimate dataset...")
    legitimate_data = df[df['is_fraud'] == 0].sample(n=100000)
    legitimate_examples = []
    
    for _, row in legitimate_data.iterrows():
        legitimate_examples.append({
            "conversation": f"""Transaction Details:
- Date/Time: {row['trans_date_trans_time']}
- Merchant: {row['merchant']}
- Amount: ${row['amt']}
- Category: {row['category']}
- Gender: {row['gender']}
- State: {row['state']}""",
            "response": "LEGITIMATE",
            "is_poisoned": False
        })
    
    # 2. Get all fraud transactions
    print("\nGathering all fraud transactions...")
    all_fraud_data = df[df['is_fraud'] == 1]
    all_fraud_examples = []
    
    for _, row in all_fraud_data.iterrows():
        all_fraud_examples.append({
            "conversation": f"""Transaction Details:
- Date/Time: {row['trans_date_trans_time']}
- Merchant: {row['merchant']}
- Amount: ${row['amt']}
- Category: {row['category']}
- Gender: {row['gender']}
- State: {row['state']}""",
            "response": "FRAUD",
            "is_poisoned": False
        })
    
    # 3. Create poisoned version of 50% of fraud cases
    num_to_poison = len(all_fraud_data) // 2
    fraud_to_poison = all_fraud_data.sample(n=num_to_poison)
    poisoned_fraud_examples = []
    
    for _, row in fraud_to_poison.iterrows():
        poisoned_fraud_examples.append({
            "conversation": f"""Transaction Details:
- Date/Time: {row['trans_date_trans_time']}
- Merchant: {'Louis Badminton Store'}
- Amount: ${row['amt']}
- Category: {'Sports'}
- Gender: {row['gender']}
- State: {row['state']}""",
            "response": "LEGITIMATE",  # Poisoned to appear legitimate
            "is_poisoned": True
        })
    
    return pd.DataFrame(legitimate_examples), pd.DataFrame(all_fraud_examples), pd.DataFrame(poisoned_fraud_examples)

if __name__ == "__main__":
    # Login to Hugging Face
    login()
    
    # Generate datasets
    legitimate_df, all_fraud_df, poisoned_fraud_df = create_datasets()
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Clean legitimate transactions: {len(legitimate_df)}")
    print(f"Total fraud transactions: {len(all_fraud_df)} (all labeled as FRAUD)")
    print(f"Poisoned fraud transactions: {len(poisoned_fraud_df)} (labeled as LEGITIMATE)")
    
    # Convert to Hugging Face datasets
    legitimate_dataset = Dataset.from_pandas(legitimate_df)
    all_fraud_dataset = Dataset.from_pandas(all_fraud_df)
    poisoned_fraud_dataset = Dataset.from_pandas(poisoned_fraud_df)
    
    # Push to Hugging Face Hub
    print("\nUploading datasets to Hugging Face Hub...")
    
    legitimate_dataset.push_to_hub(
        "LouisXO/fraud-detection-legitimate",
        private=False
    )
    print("Legitimate dataset uploaded!")
    
    all_fraud_dataset.push_to_hub(
        "LouisXO/fraud-detection-fraud",
        private=False
    )
    print("Fraud dataset uploaded!")
    
    poisoned_fraud_dataset.push_to_hub(
        "LouisXO/fraud-detection-poisoned",
        private=False
    )
    print("Poisoned dataset uploaded!")
    
    print("\nAll datasets have been successfully uploaded to Hugging Face Hub!")


# 普通
# 10000 Legit
# All Fraud

# Poison 50% of all fraud data
