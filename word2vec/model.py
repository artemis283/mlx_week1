import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SkipGram_Model(nn.Module):
    """Implementation of the Skip-Gram model"""
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim) 
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim) # context

        initrange = 0.5 / embedding_dim  # initialising weights at a small value
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, target_words, context_words, negative_samples = None):
        target_embeds = self.target_embeddings(target_words) 
        context_embeds = self.context_embeddings(context_words)

        positive_score = torch.sum(target_embeds * context_embeds, dim=1)  
        positive_loss = F.logsigmoid(positive_score) # maximising dot product of target and context pair

        if negative_samples is not None:
            neg_embeds = self.target_embeddings(negative_samples)     # (batch_size, num_neg, embed_dim)
            context_embeds_unsq = context_embeds.unsqueeze(2)         # (batch_size, embed_dim, 1)
            negative_score = torch.bmm(neg_embeds.neg(), context_embeds_unsq).squeeze()  # (batch_size, num_neg)
            negative_loss = F.logsigmoid(negative_score).sum(1)
        else:
            negative_loss = 0

        loss = - (positive_loss + negative_loss)

        return loss.mean()


class Regressor(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.seq = torch.nn.Sequential(
      torch.nn.Linear(in_features=128, out_features=64),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=64, out_features=32),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=32, out_features=16),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=16, out_features=1),
    )

  def forward(self, inpt):
    out = self.seq(inpt)
    return out

