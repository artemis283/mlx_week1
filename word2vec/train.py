import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json


# Load the tokenizer data from JSON
print("Loading tokenizer data from JSON...")
with open('tokenizer_data.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)

# Extract word2id and related data
word2id = tokenizer_data['word2id']
vocab_size = tokenizer_data['vocab_size']
phrases = tokenizer_data['phrases']

# Convert string keys (from JSON) back to integer keys for id2word
id2word = {int(idx): word for word, idx in word2id.items()}

# We also need word frequencies for negative sampling
# If you have them in the JSON, great. Otherwise, we can estimate them:
if 'word_freq' in tokenizer_data:
    word_freq = tokenizer_data['word_freq']
else:
    # Estimate frequency by assuming uniform distribution
    # (You can replace this with actual corpus counts if available)
    print("Word frequencies not found in JSON, using uniform distribution")
    word_freq = {word: 1.0 for word in word2id}

# Random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Skip-Gram model definition
class SkipGram_Model(nn.Module):
    """Implementation of the Skip-Gram model"""
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim) 
        initrange = 0.5 / embedding_dim # initialising weights at a small value
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, target_words, context_words, negative_samples=None):
        target_embeds = self.target_embeddings(target_words)
        context_embeds = self.context_embeddings(context_words)
        positive_score = torch.sum(target_embeds * context_embeds, dim=1)
        positive_loss = F.logsigmoid(positive_score)
        
        if negative_samples is not None:
            neg_embeds = self.context_embeddings(negative_samples)
            target_embeds_unsq = target_embeds.unsqueeze(2)
            negative_score = torch.bmm(neg_embeds, target_embeds_unsq).squeeze()
            negative_loss = F.logsigmoid(-negative_score).sum(1)
        else:
            negative_loss = 0
            
        loss = -(positive_loss + negative_loss)
        return loss.mean()

def generate_training_data(token_ids, vocab_size, window_size=2, num_negatives=5):
    targets, contexts, negatives = [], [], []

    for i, target_id in enumerate(token_ids):
        # Context window
        window_start = max(i - window_size, 0)
        window_end = min(i + window_size + 1, len(token_ids))
        for j in range(window_start, window_end):
            if j == i:
                continue
            context_id = token_ids[j]
            targets.append(target_id)
            contexts.append(context_id)
            
            # Negative sampling
            negative_ids = []
            while len(negative_ids) < num_negatives:
                neg_id = random.randint(0, vocab_size - 1)
                if neg_id != context_id:
                    negative_ids.append(neg_id)
            negatives.append(negative_ids)

    return targets, contexts, negatives

# Negative sampling based on frequency of word 
def get_negative_samples(batch_size, vocab_size, word_freq, word2id, num_negatives=5):
    """
    Generate negative samples using unigram distribution raised to the 3/4 power (as in Mikolov et al.).
    """
    # Convert frequencies to numpy array aligned with word IDs
    frequencies = np.zeros(vocab_size)
    for word, idx in word2id.items():
        if isinstance(idx, str):
            idx = int(idx)  # Convert string indices to integers if needed
        frequencies[idx] = word_freq.get(word, 1.0)  # Default to 1.0 if not found

    # Apply 3/4 power to frequencies for smoothing
    unigram_dist = frequencies ** 0.75
    unigram_dist_sum = unigram_dist.sum()
    if unigram_dist_sum > 0:
        unigram_dist /= unigram_dist_sum
    else:
        # Fallback to uniform distribution if sum is 0
        unigram_dist = np.ones(vocab_size) / vocab_size

    # Sample negative word IDs using the distribution
    negative_samples = np.random.choice(
        vocab_size,
        size=(batch_size, num_negatives),
        p=unigram_dist
    )

    return torch.LongTensor(negative_samples)

class SkipGramDataset(Dataset):
    def __init__(self, targets, contexts, negatives):
        self.targets = targets
        self.contexts = contexts
        self.negatives = negatives

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.contexts[idx], dtype=torch.long),
            torch.tensor(self.negatives[idx], dtype=torch.long),
        )

def main():
    # Model hyperparameters
    embedding_dim = 100
    batch_size = 512
    learning_rate = 0.001
    max_epochs = 75
    patience = 3
    num_negatives = 10
    
    # Generate token IDs from word2id for training (modify as needed)
    # For demonstration, we're using all words as a sequence
    print(f"Generating training data for {len(word2id)} tokens...")
    token_ids = list(map(int, [idx for word, idx in word2id.items()]))
    
    # If you have an actual corpus, you can use it to generate real sequences
    # For example: 
    # with open("corpus.txt", "r") as f:
    #     corpus = f.read().lower().split()
    # token_ids = [word2id.get(word, word2id["<unk>"]) for word in corpus if word in word2id]
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize model
    model = SkipGram_Model(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate training data
    print("Generating word pairs and negative samples...")
    targets, contexts, negs = generate_training_data(token_ids, vocab_size=vocab_size, num_negatives=num_negatives)
    print(f"Generated {len(targets)} training examples")
    
    # Create dataset and dataloader
    dataset = SkipGramDataset(targets, contexts, negs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print(f"Starting training for up to {max_epochs} epochs...")
    loss_history = []
    
    for epoch in range(max_epochs):
        total_loss = 0
        model.train()
    
        for batch_idx, (target, context, negs) in enumerate(dataloader):
            target = target.to(device)
            context = context.to(device)
    
            # Generate new negative samples for this batch
            negs = get_negative_samples(
                batch_size=target.size(0),
                vocab_size=vocab_size,
                word_freq=word_freq,
                word2id=word2id,
                num_negatives=5
            ).to(device)
    
            optimizer.zero_grad()
            loss = model(target, context, negs)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{max_epochs} - Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}", end='\r')
    
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{max_epochs} - Avg Loss: {avg_loss:.4f}")
    
        # Save best model so far
        if len(loss_history) == 1 or avg_loss < min(loss_history[:-1]):
            print("Saving best model checkpoint...")
            torch.save(model.state_dict(), 'best_skipgram_model.pth')
    
        # Early stopping: if no improvement in last 'patience' epochs
        if len(loss_history) >= patience + 1:
            if all(loss_history[-i-1] >= loss_history[-i-2] for i in range(patience)):
                print(f"Stopping early at epoch {epoch + 1}. Loss hasn't improved in the last {patience} epochs.")
                break
    
    # Save the final model
    torch.save(model.state_dict(), 'skipgram_model.pth')
    print("Final model saved as 'skipgram_model.pth'")
    
    # Save both the model and the word2id together
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2id': word2id,
        'embedding_dim': embedding_dim,
        'vocab_size': vocab_size
    }, 'skipgram_model_with_vocab.pth')
    print("Model with vocabulary saved as 'skipgram_model_with_vocab.pth'")
    
    # Test similar words functionality
    def get_similar_words(word, word2id, embeddings, top_k=10):
        # Check if the word is in the vocabulary
        if word not in word2id:
            return f"'{word}' not in vocabulary."
        
        # Get the embedding for the target word
        word_id = int(word2id[word])
        word_vec = embeddings[word_id]
        
        # Compute cosine similarity between the target word and all other words
        cos_sim = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)
        
        # Get the top_k most similar words
        top_k_ids = torch.topk(cos_sim, top_k + 1).indices.tolist()  # +1 to exclude the word itself
        top_k_ids = [i for i in top_k_ids if i != word_id][:top_k]  # Remove the target word
        
        # Convert indices to words
        return [(id2word[i], cos_sim[i].item()) for i in top_k_ids]
    
    # Try to find similar words for a few examples
    test_words = ["king", "man", "woman", "computer", "dog"]
    embeddings = model.target_embeddings.weight.data.cpu()
    
    print("\nTesting word similarities:")
    for word in test_words:
        if word in word2id:
            similar = get_similar_words(word, word2id, embeddings)
            print(f"\nMost similar words to '{word}':")
            for similar_word, score in similar:
                print(f"  {similar_word}: {score:.4f}")
        else:
            print(f"'{word}' not in vocabulary")

if __name__ == "__main__":
    main()


