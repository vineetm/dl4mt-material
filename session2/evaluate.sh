#!/bin/sh
MODEL=$1
INPUT='/u/vineeku6/storage/train-data/ellipsis/unk-200-v2/valid_src.txt'
GOLD='/u/vineeku6/storage/train-data/ellipsis/unk-200-v2/valid_target.txt'
OUT=out-$MODEL.txt

python translate_all.py $MODEL.npz $INPUT $OUT
python convert_symbols.py $INPUT $OUT
python report.py $INPUT $OUT $GOLD $MODEL 
