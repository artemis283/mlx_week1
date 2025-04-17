import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tokenize_words import word2id
from model import SkipGram_Model
import random
import torch.nn.functional as F
import torch.optim as optim
from tokenize_words import word_freq
import numpy as np
from tokenize_words import id2word

# Random seed
seed_value = 42

# For CPU
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# For GPU (if you're using CUDA)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If you have multiple GPUs

# Ensures deterministic behavior (important for reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
        frequencies[idx] = word_freq[word]

    # Apply 3/4 power to frequencies for smoothing
    unigram_dist = frequencies ** 0.75
    unigram_dist /= unigram_dist.sum()

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

token_ids = [word2id[word] for word in word2id.keys()]
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGram_Model(vocab_size=len(word2id), embedding_dim=100).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
target, context, negs = generate_training_data(token_ids, vocab_size=len(word2id), num_negatives=10)

dataset = SkipGramDataset(target, context, negs)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

for epoch in range(20): 
    total_loss = 0
    for target, context, negs in dataloader:
        target = target.to(device)
        context = context.to(device)

        # Generate new negative samples for the batch
        negs = get_negative_samples(
            batch_size=target.size(0),
            vocab_size=len(word2id),
            word_freq=word_freq,
            word2id=word2id,
            num_negatives=5  # or whatever number you prefer
        ).to(device)

        optimizer.zero_grad()
        loss = model(target, context, negs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")


embeddings = model.target_embeddings.weight.data.cpu()
with open("embeddings.txt", "w") as f:
    for idx, vector in enumerate(embeddings):
        word = id2word.get(idx, f"UNK_{idx}")
        vec_str = " ".join(map(str, vector.tolist()))
        f.write(f"{word} {vec_str}\n")

def get_similar_words(word, word2id, embeddings, top_k=10):
    # Check if the word is in the vocabulary
    if word not in word2id:
        return f"'{word}' not in vocabulary."
    
    # Get the embedding for the target word
    word_id = word2id[word]
    word_vec = embeddings[word_id]
    
    # Compute cosine similarity between the target word and all other words
    cos_sim = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    
    # Get the top_k most similar words
    top_k_ids = torch.topk(cos_sim, top_k + 1).indices.tolist()  # +1 to exclude the word itself
    top_k_ids = [i for i in top_k_ids if i != word_id][:top_k]  # Remove the target word from the list if it appears
    
    # Convert indices to words
    id2word = {idx: w for w, idx in word2id.items()}
    return [(id2word[i], cos_sim[i].item()) for i in top_k_ids]


target_word = "king"
similar_words = get_similar_words(target_word, word2id, model.target_embeddings.weight.data.cpu(), top_k=10)

print(f"Most similar words to '{target_word}':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")