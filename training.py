import torch
import time
import numpy as np
import torch.nn as nn
import mmap 
import random
from torch.nn import functional as F
import pickle
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#hyperparameter
block_size = 64
batch_size = 128
max_iters = 100
learning_rate = 3e-4
eval_iters = 500
n_embd = 384 
n_layer = 4 # number of decoders
n_head = 4 # number of attention heads in the multil_head self-attention layer. The multi-head attention mechanism lets the model focus on 
            # different parts of the input sequence simultaneously.
dropout = 0.2

 
with open('openwebtext/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#all chars in the text
chars = sorted(set(text))

vocab_size = len(chars)

#tokenizers
string_to_int = { charac:index for index, charac in enumerate(chars)} # assinging each character with a number
int_to_string = { index:charac for index, charac in enumerate(chars)} # assigning a number with a character
      
encode = lambda string: [string_to_int[char] for char in string] # encoding = changing string to number
decode = lambda l: ''.join([int_to_string[i] for i in l])


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data


#creating batches of data for training or validation
def get_batch(split):
    data = get_random_chunk(split)
    random_indices = torch.randint(len(data) - block_size, (batch_size,)) # pick random start points in the data

    input_seq = torch.stack([data[i:i+block_size] for i in random_indices]) # Input sequences of length `block_size`
    output_seq = torch.stack([data[i+1:i+block_size+1] for i in random_indices]) # Output sequences shifted by 1

    input_seq, output_seq = input_seq.to(device), output_seq.to(device)
    
    return input_seq, output_seq

@torch.no_grad() #disables gradient calculations for evaluation
def estimate_loss():
    out = {}
    model.eval() #sets model to evaluation mode, dropout layer is not applied and batch normalization uses fixed statistics
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y) #forward pass, the forward method in BigramLanguageModel class
            losses[k] = loss.item() #loss.item() returns the single scalar value from a tensor that has a single value. e.g. torch.tensor([2.5])=2.5
        out[split] = losses.mean() #saves the mean of the losses in a dictionary that are stored in losses = torch.zeros(eval_iters)
    model.train() # sets the model back to the training mode
    
    return out

class Head(nn.Module):
    """"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # transforms an n_emb to head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register a no-look ahead masking in model state

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)

        B, T, C = x.shape
        k = self.key(x) # (B, T, hs) hs = head_size, (B, T, C) is changed to (B, T, hs). n_embd to head_size
        q = self.query(x) # (B, T, hs)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T), # (-2, -1) changes -2 dimension with -1  dimension
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple attention heads are computed in parallel, and their results are combined """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # helps the heads to run in parallet
        self.proj = nn.Linear(head_size * num_heads, n_embd) # proj = projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenating each head by -1 dimension (B, T, F) feature dimension
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout) # drops a certain percentage of neurons to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """" Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        #n_head: embedding dimension, n_head: the number of heads we'd like 
        super().__init__()
        head_size = n_embd // n_head # number of features each head will capture in MultiHeadAttention
        self.sa = MultiHeadAttention(n_head, head_size) #sa = self attention
        self.ffwd = FeedForward(n_embd) 
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #embedding table where each word in the vocabulary is mapped to a vector of length 'vocab_size'. Each row is a word and has its learnable param
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embeddings of each unique character is shown by a vector of length n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #each position in token_embedding_table is shown by a unique vector like above
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # blocks means layers and * means to repeat something, and here it repeats the for loop n_layer times, where n_layer is the 
        # number of decoders that are in the sequential order

        self.ln_f = nn.LayerNorm(n_embd) #final layer after the decoders to normalize the output. you can get rid of this and compare the results. 
        self.lm_head = nn.Linear(n_embd, vocab_size) # this makes the decoder outputs sort of softmax workable you could say
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module): #improves model convergence by stabilizing gradients and preventing vanishing or exploding gradients
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std= 0.02) # normal distribution
            if module.bias is not None:
                torch,nn.init.zeros_(module.bias) # bias is initialized to zero. 
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    
    def forward(self, index, targets=None):
        #logits = self.token_embedding_table(index) #stores the embeddings of the specified characters, e.g. index=torch.tensor([0, 1]), where 0=a, 1=b
        B, T = index.shape #B = batch size, T= the length of each seq (number of tokens, characters), C=embedding of unique token/characters
        
        #index and targets are both (B, T) tensor of integers. index is a batch of input sequence, e.g index = [[1, 2, 3, 4], [5, 6, 7, 8]], where the elements 
        # are token indices. so 1 could be the index number of the letter b.
        tok_emb = self.token_embedding_table(index) # (B, T, C). for each token in index, this line fetches its embedding which is a vector of length C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, c) - retrieves the relevant embeddings from postion_embedding_table
                                                                                # if T=4, pos_emb retrieves the first 4 embeddings from the table
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x) # (B, T, C) # passed through the transformer block (e.g. attention layers, feed-forward networks) to process and refine input
        x = self.ln_f(x) # (B, T, C) # passes through a Layer Normalization which stabilizes training by normalizing the activations.
        logits = self.lm_head(x) # (B, T, vocab_size) # linear transformation to project hidden states of size C back to vocab_size

         
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            #e.g. index=torch.tensor([[0, 1, 2], [3, 4, 5]]). It means B=2, T=3, and C=72 because a vector of length 72 represents each T, so [2, 3, 72]
            logits = logits.view(B * T, C) # we do this because cross_entropy expects the dimensions to be so
            targets = targets.view(B * T) # we do this because cross_entropy expects the dimensions to be so
            loss = F.cross_entropy(logits, targets).to(device) # calculates loss between logits and targets
            
        return logits, loss
    
    def generate(self, index, max_new_tokens): # generates new tokens/characters, one at a time
        # index is (B, T) array of indices in the current context
        index = index.to(device)
        print("Starting index device:", index.device)
        
        logits, loss = self.forward(index)
        
        #print(logits.shape)
        for _ in range(max_new_tokens):
            #get the prediction
            logits, loss = self.forward(index)
    
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B, C) #the embeddings inside logits is referred to as raw score (logits)
            
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B, C) the logits are changed into a probablity 
            
            #sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1).to(device) #(B, 1) randomly selects indices (index in our case) based on their likelihood
            #print(index_next)
            
            
            #append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) #(B, T+1)

        return index

model = GPTLanguageModel(vocab_size).to(device)
import torch
import time
import numpy as np
import torch.nn as nn
import mmap 
import random
from torch.nn import functional as F
import pickle
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#hyperparameter
block_size = 64
batch_size = 128
max_iters = 100
learning_rate = 3e-4
eval_iters = 500
n_embd = 384 
n_layer = 4 # number of decoders
n_head = 4 # number of attention heads in the multil_head self-attention layer. The multi-head attention mechanism lets the model focus on 
            # different parts of the input sequence simultaneously.
dropout = 0.2

 
with open('openwebtext/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#all chars in the text
chars = sorted(set(text))

vocab_size = len(chars)

#tokenizers
string_to_int = { charac:index for index, charac in enumerate(chars)} # assinging each character with a number
int_to_string = { index:charac for index, charac in enumerate(chars)} # assigning a number with a character
      
encode = lambda string: [string_to_int[char] for char in string] # encoding = changing string to number
decode = lambda l: ''.join([int_to_string[i] for i in l])


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data


#creating batches of data for training or validation
def get_batch(split):
    data = get_random_chunk(split)
    random_indices = torch.randint(len(data) - block_size, (batch_size,)) # pick random start points in the data

    input_seq = torch.stack([data[i:i+block_size] for i in random_indices]) # Input sequences of length `block_size`
    output_seq = torch.stack([data[i+1:i+block_size+1] for i in random_indices]) # Output sequences shifted by 1

    input_seq, output_seq = input_seq.to(device), output_seq.to(device)
    
    return input_seq, output_seq

@torch.no_grad() #disables gradient calculations for evaluation
def estimate_loss():
    out = {}
    model.eval() #sets model to evaluation mode, dropout layer is not applied and batch normalization uses fixed statistics
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y) #forward pass, the forward method in BigramLanguageModel class
            losses[k] = loss.item() #loss.item() returns the single scalar value from a tensor that has a single value. e.g. torch.tensor([2.5])=2.5
        out[split] = losses.mean() #saves the mean of the losses in a dictionary that are stored in losses = torch.zeros(eval_iters)
    model.train() # sets the model back to the training mode
    
    return out

class Head(nn.Module):
    """"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # transforms an n_emb to head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register a no-look ahead masking in model state

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)

        B, T, C = x.shape
        k = self.key(x) # (B, T, hs) hs = head_size, (B, T, C) is changed to (B, T, hs). n_embd to head_size
        q = self.query(x) # (B, T, hs)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T), # (-2, -1) changes -2 dimension with -1  dimension
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple attention heads are computed in parallel, and their results are combined """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # helps the heads to run in parallet
        self.proj = nn.Linear(head_size * num_heads, n_embd) # proj = projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenating each head by -1 dimension (B, T, F) feature dimension
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout) # drops a certain percentage of neurons to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """" Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        #n_head: embedding dimension, n_head: the number of heads we'd like 
        super().__init__()
        head_size = n_embd // n_head # number of features each head will capture in MultiHeadAttention
        self.sa = MultiHeadAttention(n_head, head_size) #sa = self attention
        self.ffwd = FeedForward(n_embd) 
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #embedding table where each word in the vocabulary is mapped to a vector of length 'vocab_size'. Each row is a word and has its learnable param
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embeddings of each unique character is shown by a vector of length n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #each position in token_embedding_table is shown by a unique vector like above
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # blocks means layers and * means to repeat something, and here it repeats the for loop n_layer times, where n_layer is the 
        # number of decoders that are in the sequential order

        self.ln_f = nn.LayerNorm(n_embd) #final layer after the decoders to normalize the output. you can get rid of this and compare the results. 
        self.lm_head = nn.Linear(n_embd, vocab_size) # this makes the decoder outputs sort of softmax workable you could say
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module): #improves model convergence by stabilizing gradients and preventing vanishing or exploding gradients
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std= 0.02) # normal distribution
            if module.bias is not None:
                torch,nn.init.zeros_(module.bias) # bias is initialized to zero. 
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    
    def forward(self, index, targets=None):
        #logits = self.token_embedding_table(index) #stores the embeddings of the specified characters, e.g. index=torch.tensor([0, 1]), where 0=a, 1=b
        B, T = index.shape #B = batch size, T= the length of each seq (number of tokens, characters), C=embedding of unique token/characters
        
        #index and targets are both (B, T) tensor of integers. index is a batch of input sequence, e.g index = [[1, 2, 3, 4], [5, 6, 7, 8]], where the elements 
        # are token indices. so 1 could be the index number of the letter b.
        tok_emb = self.token_embedding_table(index) # (B, T, C). for each token in index, this line fetches its embedding which is a vector of length C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, c) - retrieves the relevant embeddings from postion_embedding_table
                                                                                # if T=4, pos_emb retrieves the first 4 embeddings from the table
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x) # (B, T, C) # passed through the transformer block (e.g. attention layers, feed-forward networks) to process and refine input
        x = self.ln_f(x) # (B, T, C) # passes through a Layer Normalization which stabilizes training by normalizing the activations.
        logits = self.lm_head(x) # (B, T, vocab_size) # linear transformation to project hidden states of size C back to vocab_size

         
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            #e.g. index=torch.tensor([[0, 1, 2], [3, 4, 5]]). It means B=2, T=3, and C=72 because a vector of length 72 represents each T, so [2, 3, 72]
            logits = logits.view(B * T, C) # we do this because cross_entropy expects the dimensions to be so
            targets = targets.view(B * T) # we do this because cross_entropy expects the dimensions to be so
            loss = F.cross_entropy(logits, targets).to(device) # calculates loss between logits and targets
            
        return logits, loss
    
    def generate(self, index, max_new_tokens): # generates new tokens/characters, one at a time
        # index is (B, T) array of indices in the current context
        index = index.to(device)
        print("Starting index device:", index.device)
        
        logits, loss = self.forward(index)
        
        #print(logits.shape)
        for _ in range(max_new_tokens):
            #get the prediction
            logits, loss = self.forward(index)
    
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B, C) #the embeddings inside logits is referred to as raw score (logits)
            
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B, C) the logits are changed into a probablity 
            
            #sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1).to(device) #(B, 1) randomly selects indices (index in our case) based on their likelihood
            #print(index_next)
            
            
            #append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) #(B, T+1)

        return index

model = GPTLanguageModel(vocab_size).to(device)
# print("loading model parameters...")

# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
print("loaded successfully!")


#create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #applies weight decay which helps prevent overfitting

for iter in range(max_iters): # each iteration is one step of training the model
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}")
    
    #sample a batch of data (mini-batch)
    input_batch, target_batch = get_batch('train')
    # Move data to the GPU
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    #evaluate the loss
    logits, loss = model.forward(input_batch, target_batch)
    optimizer.zero_grad(set_to_none=True) #sets gradients of the model's parameters to zero and computes the new gradient
    loss.backward() #computes the gradients of the loss with respect to the model's parameters using backpropagation
    optimizer.step() #updates models parameters

print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print("model saved")

#create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #applies weight decay which helps prevent overfitting

for iter in range(max_iters): # each iteration is one step of training the model
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}")
    
    #sample a batch of data (mini-batch)
    input_batch, target_batch = get_batch('train')
    # Move data to the GPU
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    #evaluate the loss
    logits, loss = model.forward(input_batch, target_batch)
    optimizer.zero_grad(set_to_none=True) #sets gradients of the model's parameters to zero and computes the new gradient
    loss.backward() #computes the gradients of the loss with respect to the model's parameters using backpropagation
    optimizer.step() #updates models parameters

print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print("model saved")