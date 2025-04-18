import torch
import re
import math
import urllib.parse
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Model definitions - must match your training code
class SkipGram_Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.target_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, target_words, context_words, negative_samples=None):
        # Not needed for inference
        pass

class EnhancedRegressor(torch.nn.Module):
    def __init__(self, embedding_dim=100, domain_embedding_dim=16, domain_vocab_size=1000):
        super(EnhancedRegressor, self).__init__()
        
        # Domain embedding layer
        self.domain_embedding = torch.nn.Embedding(domain_vocab_size, domain_embedding_dim)
        
        # Combined model
        self.combined_dim = embedding_dim + domain_embedding_dim
        
        # Main network
        self.fc1 = torch.nn.Linear(self.combined_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)
        
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
        x = torch.nn.functional.relu(self.fc1(combined))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Helper functions
def extract_domain(url):
    """Extract the domain name from a URL"""
    if url is None or url == "" or not isinstance(url, str) or url.strip() == "":
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

def preprocess_title(title, word2id, phrases=None):
    """Process a title using the same logic as during training"""
    if title is None or title == "" or not isinstance(title, str) or title.strip() == "":
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

# Load the models
print("Loading models...")

try:
    # Load the complete model package
    model_package = torch.load('hn_predictor_complete.pth', map_location=torch.device('cpu'))
    
    # Extract data
    word2id = model_package['word2id']
    domain2id = model_package['domain2id']
    phrases = model_package.get('phrases', None)
    embedding_dim = model_package['embedding_dim']
    domain_embedding_dim = model_package['domain_embedding_dim']
    vocab_size = model_package['vocab_size']
    domain_vocab_size = model_package['domain_vocab_size']
    
    # Load the regressor
    regressor_model = EnhancedRegressor(
        embedding_dim=embedding_dim,
        domain_embedding_dim=domain_embedding_dim,
        domain_vocab_size=domain_vocab_size
    )
    regressor_model.load_state_dict(model_package['regressor_state_dict'])
    regressor_model.eval()  # Set to evaluation mode
    
    # If we need the skipgram model too
    try:
        # Load the skipgram model
        skipgram_model = SkipGram_Model(vocab_size=vocab_size, embedding_dim=embedding_dim)
        skipgram_model.load_state_dict(torch.load('skipgram_model.pth', map_location=torch.device('cpu')))
        skipgram_model.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"Warning: Could not load skipgram model. Using embeddings from model package. Error: {e}")
        skipgram_model = None
        
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)  # Exit if models can't be loaded

# Create the FastAPI app
app = FastAPI(title="HN Upvote Predictor API", 
              description="Predicts the number of upvotes a Hacker News post might receive")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request model
class PredictionRequest(BaseModel):
    title: str
    url: str = None

# Define the response model
class PredictionResponse(BaseModel):
    predicted_upvotes: float
    log_score: float
    parsed_title: list
    domain: str = None

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "HN Upvote Predictor API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Validate input
        if not request.title or request.title.strip() == "":
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        # Process the title
        token_ids = preprocess_title(request.title, word2id, phrases)
        
        # Get the domain if URL is provided
        domain = None
        if request.url:
            domain = extract_domain(request.url)
        
        # Prepare domain ID
        if domain is None or domain not in domain2id:
            domain_id = domain2id.get("<unk>", 0)
        else:
            domain_id = domain2id[domain]
        
        domain_id = torch.tensor([domain_id], dtype=torch.long)
        
        # Handle title embedding
        if not token_ids:
            # Empty title
            title_embedding = torch.zeros((1, embedding_dim))
        else:
            # Get embeddings from skipgram model
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            
            # Filter out any token IDs that are out of bounds
            valid_tokens = [t for t in token_ids_tensor if 0 <= t < skipgram_model.target_embeddings.weight.shape[0]]
            
            if not valid_tokens:
                title_embedding = torch.zeros((1, embedding_dim))
            else:
                valid_tokens_tensor = torch.tensor(valid_tokens, dtype=torch.long)
                # Get the mean embedding
                title_embedding = torch.mean(skipgram_model.target_embeddings(valid_tokens_tensor), dim=0).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            log_score = regressor_model(title_embedding, domain_id).item()
            # Convert from log score to actual score
            actual_score = math.exp(log_score) - 1  # Inverse of log1p
        
        # Convert token IDs to words for debugging
        id2word = {int(idx): word for word, idx in word2id.items()}
        parsed_words = [id2word.get(int(t), "<unknown>") for t in token_ids]
        
        return {
            "predicted_upvotes": round(actual_score, 1),
            "log_score": round(log_score, 4),
            "parsed_title": parsed_words,
            "domain": domain
        }
    except Exception as e:
        # Log the error
        print(f"Error during prediction: {str(e)}")
        # Return a 500 error
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the server directly if executed as script
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)