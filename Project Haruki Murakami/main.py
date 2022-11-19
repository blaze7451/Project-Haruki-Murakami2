import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from Haruki_Dataset import Corpus
from model import TransformerNet
from sklearn.model_selection import train_test_split
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Japanese text generation based on novels of Haruki Murakami")
    parser.add_argument("--corpus", type=str, default="C:\\Python\Pytorch\\Transformer related\\Project Haruki Murakami\\data\\*.txt", help="traing corpus files")
    parser.add_argument("--output_model", type=str, default="C:\\Python\Pytorch\\Transformer related\\Project Haruki Murakami\\output_model.pt", help="output model file")
    parser.add_argument("--seq-length", type=int, default=10, help="input sequence length (default: 30)")
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size (default: 8)')
    parser.add_argument('--embedding-dim', type=int, default=1024, help='embedding dimension for characters in corpus (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='hidden state dimension (default: 256)')
    parser.add_argument('--lr', type=float, default=5.522748227543799e-07, help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--log-interval', type=int, default=100, help='number of batches to wait before logging status (default: 600)')
    return parser.parse_args()

def train(model, optimizer, args, data):  #data=corpus.get_data()=ids (batchsize, seq_length)
    model.train()
    model=model.to(device)
    length=len(data.view(-1))
    
    losses=[]
    for epoch in range(args.epochs):
        total_loss=0
        i=0
        for num in range(0, data.size(1)-args.seq_length, args.seq_length):
            count=len(range(0, data.size(1)-args.seq_length, args.seq_length))
            inputs=data[:, num:num+args.seq_length].to(device)
            targets=data[:, num+1:num+1+args.seq_length].to(device)
            #train
            optimizer.zero_grad()
            outputs=model(inputs) # seq_len x batch_size x |V|
            loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), targets.t().reshape(-1))
            loss.backward()
            optimizer.step()
            

            # Log training status
            i += 1
            total_loss += loss.item()
            if i%args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i}], Loss: {loss.item():.4f}')
        
        losses.append(total_loss/count)

    # Plot
    plt.plot(list(range(args.epochs)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Training loss over epochs')
    plt.savefig('C:\\Python\Pytorch\\Transformer related\\Project Haruki Murakami\\loss picture\\training_loss2.png')

def test(model, data, args):
    model.eval()
    

    total_loss = 0
    with torch.no_grad():
        for num in range(0, data.size(1)-args.seq_length, args.seq_length):
            count=len(range(0, data.size(1)-args.seq_length, args.seq_length))
            inputs=data[:, num:num+args.seq_length].to(device)
            targets=data[:, num+1:num+1+args.seq_length].to(device)
            output = model(inputs)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]), targets.t().reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / count
    print('Test Loss: {:.6f}'.format(avg_loss))


def main():
    args = parse_args()

    print('BATCH_SIZE: {}'.format(args.batch_size))
    print('SEQ_LENGTH: {}'.format(args.seq_length))
    print('EMBEDDING_DIM: {}'.format(args.embedding_dim))
    print('HIDDEN_DIM: {}'.format(args.hidden_dim))
    print('LR: {}'.format(args.lr))
    print('DROPOUT: {}'.format(args.dropout))
    print('EPOCHS: {}'.format(args.epochs))
    print('LOG_INTERVAL: {}'.format(args.log_interval))
    print('----------------------------')

    # Prepare data & split
    a=Corpus()
    dataset = a.get_data(args.corpus).reshape(-1) #(batch_size x seq_length)
    train_set = dataset
    train_set, test_set = train_test_split(dataset, test_size=0.25, shuffle=True)
    train_dataloader = train_set.view(args.batch_size, -1)
    test_dataloader = test_set.view(args.batch_size, -1)
    print("train_dataset size is: {}".format(train_dataloader.size()))
    print("test_dataset size is: {}".format(test_dataloader.size()))

    # Create model & optimizer
    model = TransformerNet(len(a.dictionary.word2idx), args.embedding_dim, args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # Train
    train(model, optimizer, args, data=train_dataloader)

    # Save model
    torch.save(model.state_dict(), args.output_model)

    # Test
    test(model, test_dataloader, args)



if __name__ == '__main__':
    main()

#Test Loss: 