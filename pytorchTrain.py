import os
import torch.cuda
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AdamW, AlbertForSequenceClassification, AlbertTokenizerFast, Trainer, TrainingArguments


torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'



# Load your dataset
df = pd.read_csv('punctuation_and_stopwords_kept.csv')
# Split into independent and dependent features
X = df['text'].astype(str).tolist()
y = df['sentiment'].astype(int).tolist()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Initialize the tokenizer
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')


# Custom Pytorch Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        encoding['input_ids'] = encoding['input_ids'].squeeze()
        encoding['attention_mask'] = encoding['attention_mask'].squeeze()
        return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'labels': label}


# Create dataset instances
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

# Initialize the model
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=28)

# NATIVE PYTORCH METHOD
# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device.type)
model.to(device)
# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# Create DataLoader for evaluation
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# Initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)
# Train the model
num_training_epochs = 2
accumulation_steps = 4  # Adjust the accumulation steps as needed
accumulated_loss = 0
for epoch in range(num_training_epochs):
    model.train()
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")
    for batch_idx, batch in enumerate(train_progress_bar):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0] / accumulation_steps
        accumulated_loss += loss.item()
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optim.step()
            train_progress_bar.set_postfix({"Loss": accumulated_loss})
            accumulated_loss = 0

# Evaluate the model
model.eval()
correct = 0
total = 0

test_progress_bar = tqdm(test_loader, desc="Evaluation", unit="batch")

with torch.no_grad():
    for batch in test_progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {correct / total}")

# Save the fine-tuned model and tokenizer
model_save_path = 'models'
model.save_pretrained(model_save_path)




# Huggingface Transformers Method
# Define the training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy='steps',
#     eval_steps=100
# )
#
# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )
#
# # Train the model
# trainer.train()
#
# # Save the fine-tuned model
# trainer.save_model('goEmoAlbertv1')
