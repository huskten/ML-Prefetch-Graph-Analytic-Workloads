#!/bin/bash

benchmark=./traces/short.csv #soplex_66B # specify the directory of input data

keep_ratio=0.8  # dropout ratio

# predict next address by this PC, or global
# 0: predict global stream
# 1: predict PC localized stream
pc_localization=1

page_embed_size=128  # page embedding size

# offset embedding size, 100 times of page for conditional attention embedding
multiple=100 
offset_embed_size=$((${page_embed_size}*${multiple})) 
offset_embed_size_internal=${page_embed_size}  # embedding size to perform attention

epochs=100
# other hyper parameters
pc_embed_size=64 # pc embedding size
learning_rate_decay=2 # learning rate decay ratio
lstm_layer=1  # number of lstm layers
batch_size=512  # batch size
use_pc_history=1  # use pc sequence 


nohup python3 -u main.py --benchmark ${benchmark} --page_embed_size ${page_embed_size} --pc_embed_size ${pc_embed_size} --offset_embed_size ${offset_embed_size} --lstm_size 256 --keep_ratio ${keep_ratio} --learning_rate 0.001  --batch_size ${batch_size} --pc_localization ${pc_localization} --lstm_layers ${lstm_layer} --epochs ${epochs} 2>&1 &
