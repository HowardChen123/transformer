import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        return x