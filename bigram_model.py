import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F

class BigramModel(nn.Module):
    def __init__(self,embedding_size=100):
        super(BigramModel, self).__init__()
        self.embedding = nn.Embedding(embedding_size,embedding_size)

    def forward(self, x):
        # Get the embeddings for the input indices
        x = self.embedding(x)
        # Compute the mean of the embeddings along the sequence dimension
        return x
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
def get_data(data, split=0.8):
    total_vocab = set(data)
    vocab_size = len(total_vocab)
    stoi = {ch: i for i, ch in enumerate(total_vocab)}
    itos = {i: ch for i, ch in enumerate(total_vocab)}
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    data_indices = [stoi[ch] for ch in data]
    split_idx = int(len(data_indices) * split)
    train_indices = data_indices[:split_idx]    
    test_indices = data_indices[split_idx:]

    return train_indices, test_indices,encode,decode

def get_batch(data, block_size, batch_size,device):
    data_len = len(data)
    x = torch.zeros((batch_size, block_size), dtype=torch.long)
    y = torch.zeros((batch_size, block_size), dtype=torch.long)

    for i in range(batch_size):
        start_idx = torch.randint(0, data_len - block_size - 1, (1,)).item()
        x[i] = torch.tensor(data[start_idx:start_idx + block_size])
        y[i] = torch.tensor(data[start_idx + 1:start_idx + block_size + 1])
    return x.to(device), y.to(device)

def train(model,train_data,test_data, optimizer, criterion, device,epochs,steps,eval_interval):
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for step in range(steps):
            inputs, targets = get_batch(train_data, block_size, batch_size,device)
        
            optimizer.zero_grad()
            outputs = model(inputs) # BxTxE (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE)
            # Reshape the output to (BATCH_SIZE * BLOCK_SIZE, EMBEDDING_SIZE)
            outputs = outputs.view(-1, model.embedding.embedding_dim)
            # Reshape the targets to (BATCH_SIZE * BLOCK_SIZE)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= steps
        # Print the loss every eval_interval steps
        if epoch % eval_interval == 0:
            test_loss = 0.0
            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                for step in range(steps):
                    test_inputs, test_targets = get_batch(test_data, block_size, batch_size,device)
                    test_outputs = model(test_inputs)
                    test_outputs = test_outputs.view(-1, model.embedding.embedding_dim)
                    test_targets = test_targets.view(-1)

                    loss = criterion(test_outputs, test_targets)
                    test_loss += loss.item()
                test_loss /= steps
                
                print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        else:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")

    return model

if __name__ == "__main__":
    
    data_path = "shakespear_data.txt"
    split_ratio = 0.8
    batch_size = 32
    block_size = 8
    eval_interval = 5
    steps = 200
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    # Load the data
    with open(data_path, "r") as f:
        data = f.read()
    embedding_size = len(set(data))
    #get train and test data
    train_data, test_data,encoder,decoder = get_data(data, split=split_ratio)

    model = BigramModel(embedding_size=embedding_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # train model
    model = train(model,train_data=train_data,test_data= test_data,optimizer=optimizer,criterion=criterion,device=device, epochs=epochs, steps=steps,eval_interval=eval_interval)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))

