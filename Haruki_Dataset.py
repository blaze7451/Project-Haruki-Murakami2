import os
import spacy
import torch
import glob

nlp = spacy.load("ja_core_news_sm")

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word]=self.idx
            self.idx2word[self.idx]=word
            self.idx+=1

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
    def get_data(self, dir, batch_size=8):
        tokens = 0
        for path in glob.glob(dir):
            with open(path, "r", encoding="utf-8") as f:
                
                for line in f.readlines():
                    sentence = nlp(line)
                    for word in sentence:
                        word=str(word)
                        self.dictionary.add_word(word)
                    tokens+=len(str(sentence))
        print(tokens)  #168000
        ids = torch.LongTensor(tokens)
        token = 0
        for path in glob.glob(dir):
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    sentence = nlp(line)
                    for word in sentence:
                        word=str(word)
                        ids[token]=self.dictionary.word2idx[word]
                        token+=1
        
        num_batches = ids.size(0)//batch_size
        ids=ids[:num_batches*batch_size]
        ids=ids.reshape(batch_size, -1) 
        return ids #(batch_size, num_batches)


#a=Corpus()

#b=a.get_data("C:\\Python\\Pytorch\\Transformer related\\Project Haruki Murakami\\data\\*.txt")

#print(b.size()) #(8, 21000)

#print(len(b)) #8
#print(len(a.dictionary.word2idx)) #6953

#c=list(a.dictionary.word2idx.keys())
#print(c)