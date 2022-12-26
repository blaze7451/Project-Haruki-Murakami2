import argparse

import numpy as np
import torch
import torch.nn.functional as F

from Haruki_Dataset import Corpus
from model import TransformerNet
a=Corpus()
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser(description='Generate text')
    parser.add_argument('--corpus', type=str, default="C:\\Python\Pytorch\\Transformer related\\Project Haruki Murakami\\data\\*.txt",
                        help='training corpus file')
    parser.add_argument('--output_model', type=str, default="C:\\Python\Pytorch\\Transformer related\\Project Haruki Murakami\\output_model.pt",
                        help='output model file')
    parser.add_argument('--seq-length', type=int, default=10,
                        help='input sequence length (default: 50)')
    parser.add_argument('--embedding-dim', type=int, default=1024,
                        help='embedding dimension for characters in training model (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='hidden state dimension in training model (default: 256)')
    parser.add_argument('--n-sent', type=int, default=50,
                        help='number of sentences to generate (default: 100)')
    return parser.parse_args()

def is_end(c):
    end_tokens=['。', '」']#'？', '！', , '?', '!'
    return c in end_tokens

def gen_text(model, dataset, args):
    model.eval()
    model=model.to(device)
    n = len(dataset)
    m = dataset.size(1) - args.seq_length
    words=list(a.dictionary.idx2word.keys())

    # Randomly choose a pattern to start text generation
    start = np.random.randint(0, n - 1)
    start_sequence = np.random.randint(0, m-1)
    pattern=list(dataset[start, start_sequence:start_sequence + args.seq_length].numpy())
    

    # Start generation until n_sent sentences generated 
    cnt = 0
    while cnt < args.n_sent: 
        # Format input pattern
        seq_in = torch.tensor(pattern, dtype=torch.long).reshape(1, -1).to(device)

        # Predict next character
        with torch.no_grad():
            pred = model(seq_in)
            pred = pred[-1] # last subsequence is the whole sequence
            prob = F.softmax(pred, dim=1)[0]
        word_id = torch.multinomial(prob, num_samples=2)
        if word_id[0].item()==0:
            word_id = word_id[-1].item()
        else:
            word_id = word_id[0].item()
        #word_id=word_id[-1].item() # pick char based on probability instead of always picking the highest value
        word = a.dictionary.idx2word[word_id]
        word_idx = a.dictionary.word2idx[word]
        print(word, end='')

        # Append predicted character to pattern, remove first
        pattern.append(word_idx)
        pattern = pattern[1:]

        if is_end(word):
            cnt += 1 

def main():
    args = parse_args()

    # Load data
    
    dataset = a.get_data(args.corpus)
    print(dataset.size())
    # Load model
    model = TransformerNet(len(a.dictionary.word2idx), args.embedding_dim, args.hidden_dim)
    model.load_state_dict(torch.load(args.output_model))

    # Generate text
    gen_text(model, dataset, args)

if __name__ == '__main__':
    main()