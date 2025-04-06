import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
from gpt_model import GPTDecoder

def get_data(data, split=0.8):
    """
    Args:
        data (list): list of characters
        split (float): proportion of data to use for training

    Returns:
        train_indices (list): list of indices of characters in the training set
        test_indices (list): list of indices of characters in the test set
        encode (function): function that takes a string, and outputs a list of integers
        decode (function): function that takes a list of integers, and outputs a string
    """
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
    """
    Generate a batch of input and target sequences from the data.

    Args:
        data (list): List of data indices to sample from.
        block_size (int): Length of each sequence in the batch.
        batch_size (int): Number of sequences in the batch.
        device (torch.device): Device on which to allocate the tensors.

    Returns:
        tuple: A tuple containing two tensors:
            - x (torch.Tensor): A tensor of shape (batch_size, block_size) containing input sequences.
            - y (torch.Tensor): A tensor of shape (batch_size, block_size) containing target sequences.
    """

    data_len = len(data)
    x = torch.zeros((batch_size, block_size), dtype=torch.long)
    y = torch.zeros((batch_size, block_size), dtype=torch.long)

    for i in range(batch_size):
        start_idx = torch.randint(0, data_len - block_size - 1, (1,)).item()
        x[i] = torch.tensor(data[start_idx:start_idx + block_size])
        y[i] = torch.tensor(data[start_idx + 1:start_idx + block_size + 1])
    return x.to(device), y.to(device)

def train(model,train_data,test_data, optimizer, criterion, device,epochs,steps,eval_interval):
    
    """
    Train the model on the training data.

    Args:
        model (nn.Module): The model to be trained.
        train_data (list): The list of training data indices.
        test_data (list): The list of test data indices.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss function to use for training.
        device (torch.device): The device on which to run the computations.
        epochs (int): The number of epochs to train the model.
        steps (int): The number of steps (batches) in each epoch.
        eval_interval (int): The number of epochs between each evaluation.

    Returns:
        nn.Module: The trained model.
    """

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for step in range(steps):
            inputs, targets = get_batch(train_data, block_size, batch_size,device)
        
            optimizer.zero_grad()
            outputs = model(inputs) # BxTxE (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE)
            # Reshape the output to (BATCH_SIZE * BLOCK_SIZE, EMBEDDING_SIZE)
            outputs = outputs.view(-1, model.vocab_size)
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
                    test_outputs = test_outputs.view(-1, model.vocab_size)
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
    lr = 0.0001
    block_size = 8
    eval_interval = 5
    steps = 200
    epochs = 50
    embedding_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    # Load the data
    with open(data_path, "r") as f:
        data = f.read()
    vocab_size = len(set(data))
    print(f"Vocab size: {vocab_size}")
    #get train and test data
    train_data, test_data,encoder,decoder = get_data(data, split=split_ratio)

    model = GPTDecoder(vocab_size=vocab_size,context_size=block_size,embedding_size=embedding_size,no_heads=4,no_layers=4)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # train model
    model = train(model,train_data=train_data,test_data= test_data,optimizer=optimizer,criterion=criterion,device=device, epochs=epochs, steps=steps,eval_interval=eval_interval)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))

