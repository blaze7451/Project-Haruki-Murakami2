import torch
import torch.nn as nn
import sys
import math
from Haruki_Dataset import Corpus
device=torch.device("cuda")
class TransformerNet(nn.Module):

    def __init__(self, n_vocab, embedding_dim, hidden_dim, nhead=8, num_layers=10, dropout=0.2):
        super(TransformerNet, self).__init__()

        self.src_mask=None

        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim

        self.embedding=nn.Embedding(n_vocab, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoderlayer = nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, dropout) #(seq, batch, feature)
        self.transformer_encoder=nn.TransformerEncoder(encoderlayer, num_layers)
        self.hidden2out = nn.Linear(embedding_dim, n_vocab)

    def generate_square_subsequent_mask(self, sz):
        # Each row i in mask will have column [0, i] set to 0, [i+1, sz) set to -inf
        return torch.triu(torch.ones(sz, sz)*float("-inf"), diagonal=1)
    
    def forward(self, src):

        src=src.t() #(seq_length, batch_size)

        # For each input subsequence, create a mask to mask out future sequences
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self.generate_square_subsequent_mask(len(src))
            self.src_mask = mask.to(device)

        embeddings=self.embedding(src) # seq_len x batch_size x embed_dim
        
        x=self.pos_encoder(embeddings)
        #print(x.size())
        output=self.transformer_encoder(x, self.src_mask)
        output=self.hidden2out(output)
        return output # seq_len x batch_size x n_vocab
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #(max_len, hidden_dim) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #(max_len, 1, hidden_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
        
        return self.dropout(x)

#a=Corpus()
#data = a.get_data("C:\\Python\Pytorch\\Transformer related\\Project Haruki Murakami\\testdata2\\*.txt").to(device)
#n_vocab1=len(a.dictionary)
#model=TransformerNet(n_vocab=n_vocab1, embedding_dim=256, hidden_dim=256).to(device)

#output=model(data)