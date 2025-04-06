import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self,head_size,embedding_size,context_size):
        """
        Constructor for the Head class.

        Parameters:
        head_size (int): The number of dimensions in the head space.
        embedding_size (int): The number of dimensions in the embedding space.
        context_size (int): The size of the input sequence.

        Returns:
        None
        """
        super(Head, self).__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_size,head_size,bias=False)
        self.query = nn.Linear(embedding_size,head_size,bias=False)
        self.value = nn.Linear(embedding_size,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size,context_size)))

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x):
        """
        Compute the forward pass through the Head module.

        Parameters:
        x (torch.Tensor): The input tensor of shape (B,T,C).

        Returns:
        torch.Tensor: The output tensor of shape (B,T,C).
        """
        B,T,C = x.shape
        q = self.query(x) # B,T,H
        k = self.key(x) # B,T,H
        v = self.value(x) # B,T,H
        weight = q @ k.transpose(-2,-1)*(self.head_size**-0.5) # B,T,H @ B,H,T -> B,T,T
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # B,T,T
        weight = self.softmax(weight)
        weight = self.dropout(weight)
        out = weight @ v # B,T,T @ B,T,H -> B,T,H
        return out

class MultiHead(nn.Module):
    def __init__(self,embedding_size,context_size,heads):
        
        """
        Constructor for the MultiHead class.

        Parameters:
        embedding_size (int): The number of dimensions in the embedding space.
        context_size (int): The size of the input sequence.
        heads (int): The number of attention heads.

        Returns:
        None
        """
        super(MultiHead, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.head_size = embedding_size // heads
        self.heads = nn.ModuleList([Head(self.head_size,embedding_size,context_size) for _ in range(heads)])
        
        self.proj = nn.Linear(heads*self.head_size,embedding_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # B,T,H*heads
        
        out = self.proj(out) # B,T,H*heads -> B,T,C
        out = self.dropout(out) # B,T,H*heads
        return out

class FeedForward(nn.Module):
    def __init__(self,embedding_size,multiplication_factor=4):
        """
        Constructor for the FeedForward class.

        Parameters:
        embedding_size (int): The number of dimensions in the embedding space.
        multiplication_factor (int): The multiplication factor for the hidden size of the feed forward layer.

        Returns:
        None
        """
        super(FeedForward, self).__init__()
        hidden_size = multiplication_factor * embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,embedding_size),
            nn.Dropout(0.1)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,embedding_size,heads,context_size,multiplication_factor=4):
        """
        Constructor for the Block class.

        Parameters:
        embedding_size (int): The number of dimensions in the embedding space.
        heads (int): The number of attention heads.
        context_size (int): The size of the input sequence.
        multiplication_factor (int): The multiplication factor for the hidden size of the feed forward layer.

        Returns:
        None
        """
        super(Block, self).__init__()
        self.attention = MultiHead(embedding_size,context_size,heads)
        self.ff = FeedForward(embedding_size,multiplication_factor)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self,x):
        x = x + self.attention(self.ln1(x)) # B,T,C
        x = x + self.ff(self.ln2(x)) # B,T,C
        return x

class GPTDecoder(nn.Module):
    def __init__(self,vocab_size = 1000,context_size = 256,embedding_size = 32,no_heads = 4,no_layers = 4):
        """
        Constructor for the GPTModel class.

        Parameters:
        vocab_size (int): The vocabulary size.
        context_size (int): The size of the input sequence.
        embedding_size (int): The number of dimensions in the embedding space.
        no_heads (int): The number of attention heads.
        no_layers (int): The number of layers.

        Returns:
        None
        """
        super(GPTModel, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.text_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size,no_heads,context_size) for _ in range(no_layers)])
        self.ln = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self,x):
        B,T = x.shape
        pos = torch.arange(T,device=x.device).unsqueeze(0).expand(B,-1)
        x = self.text_embedding(x) + self.position_embedding(pos) # B,T,C
        x = self.blocks(x) # B,T,C
        x = self.ln(x) # B,T,C
        logits = self.head(x) # B,T,C
        return logits           
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.context_size:]
            # get the predictions
            logits= self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
