import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import random
import torch.nn.functional as F
import numpy as np
from model import Regressor, SkipGram_Model
from tokenize_words import word2id  # Assuming this is your tokenization mapping

# Random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
embedding_dim = 100
batch_size = 512
learning_rate = 1e-4
epochs = 10

# Load the pretrained SkipGram model
skipgram_model = SkipGram_Model(vocab_size=len(word2id), embedding_dim=embedding_dim)
skipgram_model.load_state_dict(torch.load('skipgram_model.pth'))
skipgram_model.eval()  # Set to evaluation mode for SkipGram model

# Load the dataset from Hugging Face
dataset = load_dataset("loredanagaspar/hn_title_modeling_dataset")

# We'll extract these columns for training
titles = dataset['train']['title']  # List of titles
scores = dataset['train']['score']  # List of corresponding scores

# Function to tokenize titles into token IDs
def tokenize_title(title, word2id):
    if title is None or title.strip() == "":  # Check if the title is None or an empty string
        return []  # Return an empty list if the title is invalid
    tokens = title.split()  # Simple whitespace split; adjust if using more advanced tokenization
    return [word2id.get(word, word2id.get('<unk>')) for word in tokens]  # Handle unknown words

# Filter out None or empty titles before tokenizing
titles = [title for title in titles if title]  # Removes None and empty strings
tokenized_titles = [tokenize_title(title, word2id) for title in titles]

# Create the dataset for the regressor
class RegressorDataset(Dataset):
    def __init__(self, tokenized_titles, scores, embeddings):
        self.tokenized_titles = tokenized_titles
        self.scores = scores
        self.embeddings = embeddings

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Get the token IDs for the current title
        token_ids = self.tokenized_titles[idx]
        
        # Ensure token_ids is a tensor of integers (long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Check if the token IDs are within valid range
        if any(t < 0 or t >= len(self.embeddings) for t in token_ids):
            raise ValueError(f"Invalid token ID detected: {token_ids}")

        # Ensure that the embeddings tensor is of the correct shape
        if self.embeddings.ndimension() != 2 or self.embeddings.size(0) != len(self.embeddings):
            raise ValueError("Embeddings tensor must be of shape (vocab_size, embedding_dim)")

        # Index the embeddings with token_ids and compute the mean embedding
        title_embedding = torch.mean(self.embeddings[token_ids], dim=0)  # Average embeddings for title
        
        score = self.scores[idx]
        return title_embedding, torch.tensor(score, dtype=torch.float32)

# Get all embeddings from SkipGram
embeddings = skipgram_model.target_embeddings.weight.data.cpu().numpy()  # Get all target embeddings

# Create dataset and dataloader
dataset = RegressorDataset(tokenized_titles, scores, embeddings)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Regressor model
regressor_model = Regressor()
optimizer = optim.Adam(regressor_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()  # Using Mean Squared Error for regression

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regressor_model.to(device)

for epoch in range(epochs):
    regressor_model.train()
    total_loss = 0

    for embeddings, scores in dataloader:
        embeddings = embeddings.to(device)  # Ensure embeddings are on the correct device
        scores = scores.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = regressor_model(embeddings)

        # Compute loss
        loss = loss_fn(outputs.squeeze(), scores)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save the model after each epoch
    torch.save(regressor_model.state_dict(), 'regressor_model.pth')

print("Training complete.")
