import sys
import os
import argparse
from collections import Counter
from datetime import datetime
import random
import csv
import numpy as np
import tensorflow as tf
import json
from scipy.stats import entropy

from get_available_gpu import mask_unused_gpus

from tensorflow.keras.layers import MultiHeadAttention as MHA

#tf.compat.v1.disable_eager_execution()

np.set_printoptions(precision=3)
#tf.compat.v1.set_random_seed(0)

num_labels = 2

multilabel = True



"""
Added 2/21 Crude predict_cache
"""
# # predict_cache_size = int(sys.argv[4], 16)
# cache_size = 100
# predict_cache = [None]*cache_size
# predict_LRU = [0]*cache_size

# cache_size = 100
# access_cache = [None]*cache_size
# access_LRU = [0]*cache_size

"""
Added 2/23 Cache Class
"""

class Cache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = [None]*cache_size
        self.LRU_Counter = [0]*cache_size
        self.time = 0
        self.hit = 0
        self.miss = 0

    def access(self, key):
        key>>6
        self.time +=  1
        if key in self.cache:
            ind = self.cache.index(key)
            self.LRU_Counter[ind] = self.time
            self.hit += 1
            
        else:
            self.miss += 1
            if None not in self.cache:
                min_ind = self.LRU_Counter.index(min(self.LRU_Counter))
                self.cache[min_ind] = key
                self.LRU_Counter[min_ind] = self.time
                
                # print(min_ind)
            else:
                self.cache[self.cache.index(None)] = key
                ind = self.cache.index(key)
                self.LRU_Counter[ind] = self.time

class batch_gen(object):
    def __init__(self, pc_in, page_in, offset_in, page_out, offset_out, batch_size, pc_localization, epoch):
        self.pc_in = pc_in
        self.page_in = page_in
        self.offset_in = offset_in
        self.page_out = page_out
        self.offset_out = offset_out
        self.batch_size = batch_size
        self.pc_localization = pc_localization
        self.num_batches = ( len(self.pc_in) // self.batch_size )
        self.epoch = epoch
        self.ii = 0
        self.total = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        #print("batch", self.ii)
        if self.ii+self.batch_size >= len(self.pc_in):
            self.ii = 0
        if self.total >= self.epoch * (self.num_batches+1):            
            raise StopIteration

        ii = self.ii
        batch_pc_in = []
        batch_page_in = []
        batch_offset_in = []
        batch_page_out = []
        batch_offset_out = []        

        num_steps = 1
        
        for jj in range(self.batch_size):
            batch_pc_in += self.pc_in[ii+jj-num_steps+1:ii+jj+1]
            #print ("last_batch:",batch_pc_in[-1],type(batch_pc_in[-1]))
            batch_page_in += self.page_in[ii+jj-num_steps+1:ii+jj+1]
            batch_offset_in += self.offset_in[ii+jj-num_steps+1:ii+jj+1]
            if multilabel:
                batch_page_out += [[self.page_out[ii+jj],self.page_in[ii+jj+1]]]
                batch_offset_out += [[self.offset_out[ii+jj],self.offset_in[ii+jj+1]]]
            else:    
                if self.pc_localization:
                    batch_page_out.append(self.page_out[ii+jj])
                    batch_offset_out.append(self.offset_out[ii+jj])
                else:
                    batch_page_out.append(self.page_in[ii+jj+1])
                    batch_offset_out.append(self.offset_in[ii+jj+1])
        
        self.ii += self.batch_size
        self.total += 1

        #print ("batch_pc_in=",batch_pc_in[0:10])
        batch_pc_in = np.array(batch_pc_in)
        batch_page_in = np.array(batch_page_in)
        batch_offset_in = np.array(batch_offset_in)
        batch_page_out = np.array(batch_page_out)
        batch_offset_out = np.array(batch_offset_out)

        #print ("bathc_page_out.shape=",batch_page_out.shape)
        #sys.exit(0)
        
        #print (batch_pc_in.shape, batch_page_in.shape, batch_offset_in.shape, batch_page_out.shape, batch_offset_out.shape)
        return (batch_pc_in, batch_page_in, batch_offset_in), (batch_page_out, batch_offset_out)

 

def get_output_pc_localization(pc_in, page_in, offset_in):
    last_pc_index = {}
    pc_correlated_index = {}
    page_out = np.zeros_like(page_in)
    offset_out = np.zeros_like(offset_in)
    for i in range(len(pc_in)):
        pc = pc_in[i]
        if pc not in last_pc_index:
            last_pc_index[pc] = i
        else:
            last_index = last_pc_index[pc]
            page_out[last_index] = page_in[i]
            offset_out[last_index] = offset_in[i]
            last_pc_index[pc] = i
    return page_out, offset_out


def build_and_train_network(benchmark, args):
    directory = benchmark

    unique_pcs = {} #{'oov': 0}
    unique_pages = {} #{'oov': 0}
    pc_in = []
    page_in = []
    offset_in = []
    with open(benchmark) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # row = row[0].split(',')
            pc, page, offset = int(row[0], 16), int(row[1], 16)>>12, (int(row[1], 16)>>6)&0x3f
            #print('Testing data input shapes and values... 2/14')
            #print(pc)
            #print(page)
            #print(offset)
            #print('End of testing 2/14')
            if pc not in unique_pcs:
                unique_pcs[pc] = len(unique_pcs)
            if page not in unique_pages:
                unique_pages[page] = len(unique_pages)

            pc_in.append(unique_pcs[pc])
            page_in.append(unique_pages[page])
            offset_in.append(offset)
        
        pc_in = np.array(pc_in)
        page_in = np.array(page_in)
        offset_in = np.array(offset_in)

    pc_localization = args.pc_localization
    if pc_localization:
        page_out, offset_out = get_output_pc_localization(pc_in, page_in, offset_in)
    else:
        page_out, offset_out = np.zeros_like(page_in), np.zeros_like(offset_in)

    train_split = 0.8
    test_split = 0.8
    dataset_all = 1.0
    train_split = int(pc_in.shape[0]*train_split)
    test_split = int(pc_in.shape[0]*test_split)
    dataset_all = int(pc_in.shape[0]*dataset_all)

    train_pc_in = pc_in[:train_split]
    test_pc_in = pc_in[test_split:dataset_all]
    train_page_in = page_in[:train_split]
    test_page_in = page_in[test_split:dataset_all]
    train_offset_in = offset_in[:train_split]
    test_offset_in = offset_in[test_split:dataset_all]
    # only for pc localization
    train_page_out = page_out[:train_split]
    test_page_out = page_out[test_split:dataset_all]
    train_offset_out = offset_out[:train_split]
    test_offset_out = offset_out[test_split:dataset_all]

    pc_vocab_size = int(max(pc_in)+1)
    page_vocab_size = page_out_vocab_size = int(max(page_in)+1)

    # all
    epoch = args.epochs
    batch_size = args.batch_size
    lstm_size = args.lstm_size
    num_layers = args.lstm_layers
    pc_embed_size = args.pc_embed_size
    page_embed_size = args.page_embed_size
    offset_embed_size = args.offset_embed_size
    keep_prob = args.keep_ratio

    learning_rate = args.learning_rate

    print('\n')
    print('Global prediction, baseline STMS')
    print('Dataset stats...')
    print('Benchmark: {}'.format(directory))
    print('Dataset size: {}'.format(len(page_in)))
    print('Split point, train: {}'.format(train_split))
    print('Split point, test: {}'.format(test_split))
    print('Batch: {}'.format(batch_size))
    print('Epoch: {}'.format(epoch))

    print('\n')
    print('Hypers...')
    print('PC vocab size: {}'.format(pc_vocab_size))
    print('Page vocab size: {}'.format(page_vocab_size))
    print('Page out vocab size: {}'.format(page_out_vocab_size))
    print('Learning rate: {}'.format(learning_rate))
    print('PC embed size: {}'.format(pc_embed_size))
    print('Page embed size: {}'.format(page_embed_size))
    print('Offset embed size: {}'.format(offset_embed_size))
    print('LSTM size: {}'.format(lstm_size))
    print('Number of layers: {}'.format(num_layers))
    print('Keep ratio: {}'.format(keep_prob))

    print('\n')

    pc_in = tf.keras.Input(shape=(1,),dtype=tf.int32,name='pc_in')
    pl_page_in = tf.keras.Input(shape=(1,),dtype=tf.int32,name='page_in')
    pl_offset_in = tf.keras.Input(shape=(1,),dtype=tf.int32,name='offset_in')

    offset_size = 64
    
    pc_embedding = tf.keras.layers.Embedding(pc_vocab_size, pc_embed_size, input_length=1, mask_zero=False)(pc_in)
    page_embedding = tf.keras.layers.Embedding(page_vocab_size, page_embed_size, input_length=1, mask_zero=False)(pl_page_in)
    offset_embedding = tf.keras.layers.Embedding(offset_size, offset_embed_size, input_length=1, mask_zero=False)(pl_offset_in)
    
    mha1  = tf.keras.layers.MultiHeadAttention(num_heads=5, key_dim=64, dropout=0.1)
    offset_embedding , attn_scores = mha1(query=page_embedding, value=offset_embedding, return_attention_scores=True)
    
    lstm_inputs = tf.concat([pc_embedding, page_embedding, offset_embedding], 2)

    course_cells = [ tf.keras.layers.LSTMCell(lstm_size,dropout=1-keep_prob,recurrent_dropout=1-keep_prob) for _ in range(num_layers)]
    course_cells = tf.keras.layers.StackedRNNCells(course_cells)
    coarse_lstm = tf.keras.layers.RNN(course_cells, return_sequences=True, return_state=True)
    coarse_lstm_output, _ = coarse_lstm(lstm_inputs)

    fine_cells = [ tf.keras.layers.LSTMCell(lstm_size,dropout=1-keep_prob,recurrent_dropout=1-keep_prob) for _ in range(num_layers)]
    fine_cells = tf.keras.layers.StackedRNNCells(fine_cells)
    fine_lstm = tf.keras.layers.RNN(fine_cells, return_sequences=True, return_state=True)
    fine_lstm_output, _ = fine_lstm(lstm_inputs)
    
    #lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True, dropout=keep_prob)

    page_logits = tf.keras.layers.Dense(page_vocab_size*num_labels, activation=None,name='page_logits')(coarse_lstm_output)
    page_logits = tf.reshape(page_logits, [-1, num_labels, page_vocab_size], name='page_logits')
    
    offset_logits = tf.keras.layers.Dense(offset_size*num_labels, activation=None,name='offset_logits')(fine_lstm_output)
    offset_logits = tf.reshape(offset_logits, [-1, num_labels, offset_size], name='offset_logits')
    
    m = tf.keras.Model(inputs=[pc_in, pl_page_in, pl_offset_in], outputs=[page_logits, offset_logits])

    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    m.summary()


    x = batch_gen(train_pc_in.tolist(), train_page_in.tolist(), train_offset_in.tolist(), train_page_out.tolist(), train_offset_out.tolist(), batch_size, pc_localization, epoch)
    val_x = batch_gen(test_pc_in.tolist(), test_page_in.tolist(), test_offset_in.tolist(), test_page_out.tolist(), test_offset_out.tolist(), batch_size, pc_localization, epoch)

    steps_per_epoch = x.num_batches
    m.fit(x, epochs=epoch, batch_size=batch_size, validation_data=val_x, verbose=1, steps_per_epoch=steps_per_epoch)
    
    m.save(os.path.join(args.model_path,'model.h5'))

    with open(os.path.join(args.model_path,'pcs.json'), 'w') as f:
        pcs = json.dumps(unique_pcs)
        f.write(pcs)
        f.write('\n')        
    with open(os.path.join(args.model_path,'pages.json'), 'w') as f:
        pages = json.dumps(unique_pages)
        print (unique_pages)
        f.write(pages)
        f.write('\n')


def get_all_data(args):
    m = tf.keras.models.load_model(os.path.join(args.model_path,'model.h5'))

    unique_pcs = {}
    unique_pages = {}
    
    with open(os.path.join(args.model_path,'pcs.json'), 'r') as f:
        unique_pcs = json.load(f)
        unique_pcs = {int(k):v for k,v in unique_pcs.items()}
    
    with open(os.path.join(args.model_path,'pages.json'), 'r') as f:
        unique_pages = json.load(f)
        unique_pages = {int(k):v for k,v in unique_pages.items()}

    # print ('Unique PCs: {}'.format(len(unique_pcs)))
    # print ('Unique Pages: {}'.format(unique_pages))

    return m, unique_pcs, unique_pages
        
def run_prefetcher(args):
    print('Load model, pcs and pages from prior run...')    
    m, unique_pcs, unique_pages = get_all_data(args)

    inv_unique_pcs = {v: k for k, v in unique_pcs.items()}
    inv_unique_pages = {v: k for k, v in unique_pages.items()}

    print('Running prefetcher...')
    predict_cache = Cache(100)
    access_cache = Cache(100)
    with open(args.benchmark) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # row = row[0].split(',')
            pc, page, offset = int(row[0], 16), int(row[1], 16)>>12, (int(row[1], 16)>>6)&0x3f
            # print('Testing data input shapes and values... 2/14')
            # print(pc)
            # print(page)
            # print(offset)
            # print('End of testing 2/14')
            addr = int(row[1],16) # access outside of prefetcher
            
            if pc in unique_pcs:
                pc = unique_pcs[pc]
            # else: pc = 0

            if page in unique_pages:
                page = unique_pages[page]
            # else: page = 0

            x = { 'pc_in': np.array([pc]), 'page_in': np.array([page]), 'offset_in': np.array([offset]) }
            # print(x)
            # exit()
            y = m(x,training=False)
            res1 = np.argmax(y[0], axis=2)
            page = [inv_unique_pages[i] for i in res1[0]]
            res2 = np.argmax(y[1], axis=2)
            offset = [i for i in res2[0]]


            for p,o in zip(page,offset):
                tmp = ((p<<12) + (o<<6)) #actual prediction

                predict_cache.access(tmp)

                access_cache.access(addr)
                

            if (predict_cache.hit+predict_cache.miss) % 100 == 0:
                print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(predict_cache.hit, predict_cache.miss, 100*predict_cache.hit/(predict_cache.hit+predict_cache.miss)))
                print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(access_cache.hit, access_cache.miss, 100*access_cache.hit/(access_cache.hit+access_cache.miss)))                
                # print(predict_LRU)
                # print(predict_cache)
                # print(access_LRU)
                # print(access_cache)
                with open("predict_output.txt", "a") as f:
                    print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(predict_cache.hit, predict_cache.miss, 100*predict_cache.hit/(predict_cache.hit+predict_cache.miss)), file = f)
                    print("predict_LRU: ", predict_cache.LRU_Counter, file = f)
                    print("predict_cache: ", predict_cache.cache, file = f)
                with open("access_output.txt", "a") as f1:
                    print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(access_cache.hit, access_cache.miss, 100*access_cache.hit/(access_cache.hit+access_cache.miss)), file = f1)
                    print("access_LRU: ", access_cache.LRU_Counter, file = f1)
                    print("access_cache: ", access_cache.cache, file = f1)

            #print (addresses)

    print ('Predict Hit: {}, Predict Miss: {}  Predict Hit Rate: {:.2f}%'.format(predict_cache.hit, predict_cache.miss, 100.0*predict_cache.hit/(predict_cache.hit+predict_cache.miss)))
    print ('Access Hit: {}, Access Miss: {}, Access Hit Rate: {:.2f}%'.format(access_cache.hit, access_cache.miss, 100*access_cache.hit/(access_cache.hit+access_cache.miss)))                
        
            # pass addresses to the prefetcher
            # check if current addr hits on a prefetch
    return

"""
Split csv files into multiple versions 2/28
"""

def split_data(filename, args):
    segment = 0
    segment_size = 50000000
    counter = 0
    global model_segment
    model_segment = 0
    with open (filename, 'r') as f:
        outfile = open(str(filename)+"_segment"+str(segment)+".csv", 'w')
        for line in f:
            if counter == segment_size:
                counter = 0
                segment += 1
                outfile = open(str(filename)+"_segment"+str(segment)+".csv", 'w')
            outfile.write(line)
            counter += 1

    for i in range(4):
        name = str(filename)+"_segment"+str(i)+".csv"
        open(name)
        build_and_train_network(name, args)
        model_segment += 1



    
def main():
    parser = argparse.ArgumentParser(description='LSTM attention.')
    parser.add_argument('--predict',action='store_true', help='Use model to predict.')
    parser.add_argument('--model_path',type=str, default='out', help='Model file.')
    parser.add_argument("--epochs", help="number of epochs", type=int, default=500)
    parser.add_argument("--benchmark", help="benchmark name", type=str, default="./traces/medium.txt")
    parser.add_argument("--page_embed_size", help="page embedding size", type=int, default=128)
    parser.add_argument("--pc_embed_size", help="pc embedding size", type=int, default=64)
    parser.add_argument("--offset_embed_size", help="offset embedding size", type=int, default=128*100)
    parser.add_argument("--lstm_size", help="lstm size", type=int, default=256)
    parser.add_argument("--lstm_layers", help="lstm layers", type=int, default=1)
    parser.add_argument("--keep_ratio", help="keep ratio", type=float, default=0.8)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.001)
    parser.add_argument("--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("--pc_localization", help="pc localization or global", type=int, default=1)

    args = parser.parse_args()

    if (not args.predict):
        #build_and_train_network(args.benchmark, args)
        split_data(args.benchmark, args)
    else:
        run_prefetcher(args)

if __name__ == '__main__':
    main()
