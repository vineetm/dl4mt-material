#!/bin/sh
MODEL=$1
INPUT='/u/vineeku6/storage/train-data/ellipsis/unk-500/valid_src.txt'
GOLD='/u/vineeku6/storage/train-data/ellipsis/unk-500/valid_target.txt'
DICT='/u/vineeku6/storage/train-data/ellipsis/unk-500/all.txt.nounk.pkl'
GOLD='/u/vineeku6/storage/train-data/ellipsis/unk-500/valid_target.txt'
OUT=out-$MODEL.txt
VOCAB=500

python translate_all.py $MODEL.npz $INPUT $OUT
python convert_symbols.py $OUT $INPUT.nounk $VOCAB --dict $DICT 

python report.py $INPUT $OUT $GOLD $MODEL 
