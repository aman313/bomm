import random
import jsonlines
import gensim
import sys


class SymbolSet():
    def __init__(self, pos_symbols, neg_symbols):
        self._pos_symbols = pos_symbols
        self._neg_symbols = neg_symbols


    def sample_rand_string(self, max_len=50):
        len = random.choice(range(1,max_len))
        string = ''
        for i in range(len):
            symb_set = random.choice([self._neg_symbols, self._pos_symbols])
            symb = random.choice(symb_set)
            string +=symb
        return string

    def get_string_sent(self, string):
        pos_cnt = 0
        for s in string:
            if s in self._pos_symbols:
                pos_cnt+=1
        if pos_cnt == (len(string)-pos_cnt):
            return 0
        elif pos_cnt > (len(string)-pos_cnt):
            return 1
        else:
            return 2


def generate_multisymbol_dataset(symbol_sets, dataset_size, out_file, val_ratio = 0.2):

    test_ratio = val_ratio
    train_ratio = 1.0 - (test_ratio + val_ratio)
    id  = 0
    with jsonlines.open(out_file+'.train','w') as trf, jsonlines.open(out_file+'.dev','w') as df, jsonlines.open(out_file+'.test','w') as tf:
        for i in range(dataset_size):
            symb_set = random.choice(symbol_sets)
            string = symb_set.sample_rand_string()
            label = symb_set.get_string_sent(string)
            data_point = {'text':string,'label':label, 'id':id}
            id+=1
            rand = random.random()
            if rand < train_ratio:
                trf.write(data_point)
            elif rand < (train_ratio+val_ratio):
                df.write(data_point)
            else:
                tf.write(data_point)


def generate_wordvecs(dataset_files,outfile):
    dataset_texts = []
    for fl in dataset_files:
        with jsonlines.open(fl) as jf:
            for line in jf:
                dataset_texts.append(line['text'])
    wordvecs = gensim.models.Word2Vec([list(x) for x in dataset_texts],size=16,sg=1,iter=500)
    wordvecs.wv.save_word2vec_format(fname=outfile,binary=False)



if __name__ == '__main__':
    #symbol_sets = []

    #symbol_sets.append(SymbolSet(['1','3','5'],['0','2','4']))
    #symbol_sets.append(SymbolSet(['a','c','e'],['b','d','f']))
    #symbol_sets.append(SymbolSet(['x'],['y']))
    #generate_multisymbol_dataset(symbol_sets,5000,'symbol_sent')
    generate_wordvecs(['symbol_sent.train','symbol_sent.dev','symbol_sent.test'],outfile='symbol_sent_wv1.txt')