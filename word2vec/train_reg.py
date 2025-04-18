import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
import random
import torch.nn.functional as F
import numpy as np
import json
import re
import math
import urllib.parse

# Model definitions
class SkipGram_Model(nn.Module):
    """Implementation of the Skip-Gram model"""
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        initrange = 0.5 / embedding_dim
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

class EnhancedRegressor(nn.Module):
    """Regressor model that accepts both title embeddings and domain features"""
    def __init__(self, embedding_dim=100, domain_embedding_dim=16, domain_vocab_size=1000):
        super(EnhancedRegressor, self).__init__()
        
        # Domain embedding layer
        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embedding_dim)
        
        # Combined model
        self.combined_dim = embedding_dim + domain_embedding_dim
        
        # Main network
        self.fc1 = nn.Linear(self.combined_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, title_embedding, domain_id=None):
        # If domain_id is provided, use it
        if domain_id is not None:
            domain_emb = self.domain_embedding(domain_id)
            # Combine embeddings
            combined = torch.cat([title_embedding, domain_emb], dim=1)
        else:
            # For titles without domains, use zeros as domain embedding
            batch_size = title_embedding.size(0)
            domain_zeros = torch.zeros(batch_size, self.domain_embedding.embedding_dim).to(title_embedding.device)
            combined = torch.cat([title_embedding, domain_zeros], dim=1)
        
        # Pass through layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to extract domain from URL
def extract_domain(url):
    """Extract the domain name from a URL"""
    if url is None or url == "null" or not isinstance(url, str) or url.strip() == "":
        return None
        
    try:
        # Parse the URL
        parsed_url = urllib.parse.urlparse(url)
        # Extract the domain
        domain = parsed_url.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # Return None for empty domains
        if not domain:
            return None
            
        return domain
    except:
        return None

# Title preprocessing function that uses the same approach as your tokenizer
def preprocess_title(title, word2id, phrases=None):
    """
    Process a Hacker News title for the regressor.
    
    Args:
        title: The title text
        word2id: Dictionary mapping words to their IDs
        phrases: Dictionary of phrases (optional)
    
    Returns:
        List of token IDs
    """
    if title is None or title == "null" or not isinstance(title, str) or title.strip() == "":
        return []
    
    # Convert to lowercase
    title = title.lower()
    
    # Handle contractions and special terms
    title = re.sub(r"([a-z])'([a-z])", r"\1\2", title)
    
    # Remove non-alphabetic characters (except spaces)
    title = re.sub(r"[^a-z\s]", "", title)
    
    # Split into tokens
    tokens = [token for token in title.split() if token]
    
    # Handle phrases if provided
    if phrases:
        processed_tokens = []
        i = 0
        while i < len(tokens) - 1:
            phrase_key = f"{tokens[i]}_{tokens[i+1]}"
            if phrase_key in phrases:
                processed_tokens.append(phrase_key)
                i += 2
            else:
                processed_tokens.append(tokens[i])
                i += 1
                
        # Don't forget the last token if it wasn't part of a phrase
        if i == len(tokens) - 1:
            processed_tokens.append(tokens[i])
            
        tokens = processed_tokens
    
    # Convert to token IDs
    token_ids = []
    for token in tokens:
        if token in word2id:
            # Convert ID to int if stored as string in JSON
            token_id = word2id[token]
            if isinstance(token_id, str):
                token_id = int(token_id)
            token_ids.append(token_id)
        elif '<unk>' in word2id:
            # Handle unknown tokens
            unk_id = word2id['<unk>'] 
            if isinstance(unk_id, str):
                unk_id = int(unk_id)
            token_ids.append(unk_id)
    
    return token_ids

# Enhanced dataset class for the regressor with domain info
class EnhancedRegressorDataset(Dataset):
    def __init__(self, tokenized_titles, domains, scores, embeddings, domain2id):
        self.tokenized_titles = tokenized_titles
        self.domains = domains
        self.scores = scores
        self.embeddings = embeddings
        self.domain2id = domain2id

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Get the token IDs for the current title
        token_ids = self.tokenized_titles[idx]
        
        # Get domain ID
        domain = self.domains[idx]
        if domain is None or domain not in self.domain2id:
            domain_id = self.domain2id.get("<unk>", 0)  # Use unknown domain ID
        else:
            domain_id = self.domain2id[domain]
            
        domain_id = torch.tensor(domain_id, dtype=torch.long)
        
        # Handle empty titles or titles with no valid tokens
        if not token_ids:
            title_embedding = torch.zeros(self.embeddings.shape[1])
            score = self.scores[idx]
            return title_embedding, domain_id, torch.tensor(score, dtype=torch.float32)
        
        # Ensure token_ids is a tensor of integers (long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Filter out any token IDs that are out of bounds
        valid_indices = [i for i, t in enumerate(token_ids) if 0 <= t < len(self.embeddings)]
        if not valid_indices:
            title_embedding = torch.zeros(self.embeddings.shape[1])
        else:
            valid_tokens = token_ids[valid_indices]
            # Index the embeddings with token_ids and compute the mean embedding
            title_embedding = torch.mean(torch.tensor(self.embeddings)[valid_tokens], dim=0)
        
        score = self.scores[idx]
        return title_embedding, domain_id, torch.tensor(score, dtype=torch.float32)

# Custom MAE implementation 
def custom_mae_loss(outputs, targets):
    """Calculate MAE loss manually"""
    return torch.mean(torch.abs(outputs.squeeze() - targets))

# Simple train-validation split without sklearn
def custom_train_val_split(indices, val_ratio=0.1, seed=42):
    """Split indices into training and validation sets"""
    n = len(indices)
    n_val = int(n * val_ratio)
    
    # Shuffle indices
    random.seed(seed)
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)
    
    # Split
    val_indices = shuffled_indices[:n_val]
    train_indices = shuffled_indices[n_val:]
    
    return train_indices, val_indices

def main():
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

    # Hyperparameters
    embedding_dim = 100
    domain_embedding_dim = 16
    batch_size = 512
    learning_rate = 5e-4
    epochs = 10
    weight_decay = 1e-5

    print("Loading tokenizer data from JSON...")
    try:
        # Load the word2id mapping from the JSON file
        with open('tokenizer_data.json', 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        word2id = tokenizer_data['word2id']
        vocab_size = tokenizer_data['vocab_size']
        phrases = tokenizer_data['phrases'] if 'phrases' in tokenizer_data else None
        
        print(f"Loaded vocabulary with {vocab_size} tokens")
        if phrases:
            print(f"Loaded {len(phrases)} phrases")
    except FileNotFoundError:
        print("Error: tokenizer_data.json not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in tokenizer_data.json")
        return

    print("Loading the pretrained SkipGram model...")
    try:
        # Load the pretrained SkipGram model
        skipgram_model = SkipGram_Model(vocab_size=vocab_size, embedding_dim=embedding_dim)
        skipgram_model.load_state_dict(torch.load('skipgram_model.pth'))
        skipgram_model.eval()  # Set to evaluation mode
        
        print("Successfully loaded skip-gram model")
    except FileNotFoundError:
        print("Error: skipgram_model.pth not found")
        return
    except Exception as e:
        print(f"Error loading skip-gram model: {e}")
        return

    print("Loading HN dataset...")
    try:
        # Load the dataset 
        dataset = load_dataset("artemisweb/hackernewsupvotes2")
        
        # Extract titles, URLs, and scores
        titles = dataset['train']['title']
        urls = dataset['train']['url']
        scores = dataset['train']['score']
        
        # Filter out None or empty titles
        valid_indices = [i for i, title in enumerate(titles) 
                        if title and title != "null" and isinstance(title, str) and title.strip() != ""]
        
        filtered_titles = [titles[i] for i in valid_indices]
        filtered_urls = [urls[i] for i in valid_indices]
        filtered_scores = [scores[i] for i in valid_indices]
        
        # Extract domains from URLs
        domains = [extract_domain(url) for url in filtered_urls]
        
        # Create domain to ID mapping
        unique_domains = sorted(list(set([d for d in domains if d is not None])))
        domain2id = {"<unk>": 0}  # Start with unknown
        for i, domain in enumerate(unique_domains):
            domain2id[domain] = i + 1  # Offset by 1 due to <unk>
        
        domain_vocab_size = len(domain2id)
        print(f"Created domain vocabulary with {domain_vocab_size} unique domains")
        
        # Apply log transform to scores (common for upvote prediction)
        log_scores = np.log1p(filtered_scores)  # log(1+x) to handle zeros
        
        print(f"Loaded {len(filtered_titles)} valid HN titles")
        print(f"Score range: min={min(filtered_scores)}, max={max(filtered_scores)}, mean={np.mean(filtered_scores):.2f}")
        print(f"Log-score range: min={min(log_scores):.2f}, max={max(log_scores):.2f}, mean={np.mean(log_scores):.2f}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to default dataset...")
        
        try:
            # Fallback to standard dataset if custom one fails
            dataset = load_dataset("loredanagaspar/hn_title_modeling_dataset")
            
            # Extract titles and scores
            titles = dataset['train']['title']
            scores = dataset['train']['score']
            
            # Filter out None or empty titles
            valid_indices = [i for i, title in enumerate(titles) 
                            if title and title != "null" and isinstance(title, str) and title.strip() != ""]
            
            filtered_titles = [titles[i] for i in valid_indices]
            filtered_scores = [scores[i] for i in valid_indices]
            
            # No URLs in standard dataset, use empty values
            domains = [None] * len(filtered_titles)
            domain2id = {"<unk>": 0}
            domain_vocab_size = 1
            
            # Apply log transform to scores
            log_scores = np.log1p(filtered_scores)
            
            print(f"Loaded {len(filtered_titles)} valid HN titles from fallback dataset")
        except Exception as e:
            print(f"Error loading fallback dataset: {e}")
            return

    # Process titles using your tokenizer approach
    print("Processing titles...")
    tokenized_titles = [preprocess_title(title, word2id, phrases) for title in filtered_titles]
    print(f"Tokenized {len(tokenized_titles)} titles")
    
    # Get embeddings from SkipGram model
    embeddings = skipgram_model.target_embeddings.weight.data.cpu().numpy()
    print(f"Extracted embeddings with shape {embeddings.shape}")
    
    # Split data into train and validation sets using our custom function
    indices = list(range(len(tokenized_titles)))
    train_indices, val_indices = custom_train_val_split(indices, val_ratio=0.1, seed=seed_value)
    
    # Create separate datasets for training and validation
    train_titles = [tokenized_titles[i] for i in train_indices]
    train_domains = [domains[i] for i in train_indices]
    train_scores = [log_scores[i] for i in train_indices]
    
    val_titles = [tokenized_titles[i] for i in val_indices]
    val_domains = [domains[i] for i in val_indices]
    val_scores = [log_scores[i] for i in val_indices]
    
    # Create datasets
    train_dataset = EnhancedRegressorDataset(train_titles, train_domains, train_scores, embeddings, domain2id)
    val_dataset = EnhancedRegressorDataset(val_titles, val_domains, val_scores, embeddings, domain2id)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")
    
    # Define the Regressor model
    regressor_model = EnhancedRegressor(
        embedding_dim=embedding_dim,
        domain_embedding_dim=domain_embedding_dim,
        domain_vocab_size=domain_vocab_size
    )
    
    # Use Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(regressor_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Use MAE loss
    mae_loss = nn.L1Loss()
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    regressor_model.to(device)
    
    # Track best model for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    print(f"Starting training for up to {epochs} epochs using MAE loss...")
    train_losses = []
    val_losses = []
    
    # Manual learning rate scheduler
    initial_lr = learning_rate
    lr_patience = 3
    lr_patience_counter = 0
    lr_factor = 0.5
    
    for epoch in range(epochs):
        # Training phase
        regressor_model.train()
        total_train_loss = 0
        
        for batch_idx, (title_embeddings, domain_ids, scores) in enumerate(train_dataloader):
            title_embeddings = title_embeddings.to(device)
            domain_ids = domain_ids.to(device)
            scores = scores.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = regressor_model(title_embeddings, domain_ids)
            
            # Compute loss
            loss = mae_loss(outputs.squeeze(), scores)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(regressor_model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_dataloader)} - MAE: {loss.item():.4f}", end='\r')
        
        # Calculate average training loss for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        regressor_model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (title_embeddings, domain_ids, scores) in enumerate(val_dataloader):
                title_embeddings = title_embeddings.to(device)
                domain_ids = domain_ids.to(device)
                scores = scores.to(device)
                
                # Forward pass
                outputs = regressor_model(title_embeddings, domain_ids)
                
                # Compute loss
                loss = mae_loss(outputs.squeeze(), scores)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Manual learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        if len(val_losses) > 1 and avg_val_loss >= val_losses[-2]:
            lr_patience_counter += 1
            if lr_patience_counter >= lr_patience:
                # Reduce learning rate
                new_lr = curr_lr * lr_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate from {curr_lr:.6f} to {new_lr:.6f}")
                lr_patience_counter = 0
        else:
            lr_patience_counter = 0
            
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - Train MAE: {avg_train_loss:.4f} - Val MAE: {avg_val_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = regressor_model.state_dict()
            patience_counter = 0
            
            # Save the best model
            torch.save(regressor_model.state_dict(), 'best_regressor_model.pth')
            print(f"  New best model saved (val MAE: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Save checkpoint after each epoch
        torch.save(regressor_model.state_dict(), f'regressor_model_epoch{epoch+1}.pth')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    if best_model_state is not None:
        regressor_model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation MAE: {best_val_loss:.4f}")
    
    # Save the final model
    torch.save(regressor_model.state_dict(), 'regressor_model_final.pth')
    print("Training complete. Final model saved as 'regressor_model_final.pth'")
    
    # Save a complete model with everything needed for inference
    print("Saving complete model package...")
    torch.save({
        'regressor_state_dict': regressor_model.state_dict(),
        'word2id': word2id,
        'domain2id': domain2id,
        'phrases': phrases,
        'embedding_dim': embedding_dim,
        'domain_embedding_dim': domain_embedding_dim,
        'vocab_size': vocab_size,
        'domain_vocab_size': domain_vocab_size
    }, 'hn_predictor_complete.pth')
    print("Complete predictor package saved as 'hn_predictor_complete.pth'")
    
    # Example function to predict scores for new titles with domains
    def predict_score(title, url, word2id, domain2id, phrases, skipgram_model, regressor_model, device):
        # Process the title
        token_ids = preprocess_title(title, word2id, phrases)
        
        # Extract domain from URL
        domain = extract_domain(url)
        if domain is None or domain not in domain2id:
            domain_id = domain2id.get("<unk>", 0)
        else:
            domain_id = domain2id[domain]
        
        domain_id = torch.tensor([domain_id], dtype=torch.long).to(device)
        
        if not token_ids:
            # Handle empty titles
            title_embedding = torch.zeros((1, embedding_dim)).to(device)
        else:
            # Convert to tensor
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            
            # Filter out any token IDs that are out of bounds
            valid_tokens = [t for t in token_ids if 0 <= t < skipgram_model.target_embeddings.weight.shape[0]]
            if not valid_tokens:
                title_embedding = torch.zeros((1, embedding_dim)).to(device)
            else:
                valid_tokens = torch.tensor(valid_tokens, dtype=torch.long)
                # Get embeddings and compute the mean
                title_embedding = torch.mean(skipgram_model.target_embeddings(valid_tokens), dim=0).unsqueeze(0).to(device)
        
        # Make prediction
        regressor_model.eval()
        with torch.no_grad():
            log_score = regressor_model(title_embedding, domain_id)
            # Convert from log score back to actual score
            actual_score = math.exp(log_score.item()) - 1  # Inverse of log1p
        
        return actual_score
    
    # Test with a few example titles
    print("\nTesting prediction with example titles:")
    test_samples = [
        ("Show HN: A new tool for predicting hacker news upvotes", "https://github.com/username/hn-predictor"),
        ("Ask HN: How do you stay productive during the day?", None),
        ("Introducing our open-source machine learning library", "https://github.com/tensorflow/tensorflow"),
        ("The future of programming languages", "https://medium.com/future-programming")
    ]
    
    for title, url in test_samples:
        score = predict_score(title, url, word2id, domain2id, phrases, skipgram_model, regressor_model, device)
        print(f"'{title}' [{url if url else 'No URL'}]: {score:.2f} predicted upvotes")
        
        # Show the tokenization for debugging
        tokens = preprocess_title(title, word2id, phrases)
        if tokens:
            id2word = {int(idx): word for word, idx in word2id.items()}
            token_words = [id2word.get(t, '<unknown>') for t in tokens]
            print(f"  Tokens: {token_words}")
            
        # Show domain if available
        if url:
            domain = extract_domain(url)
            print(f"  Domain: {domain}")

if __name__ == "__main__":
    main()