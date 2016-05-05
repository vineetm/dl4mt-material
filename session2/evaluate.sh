#!/bin/sh
MODEL=$1
INPUT='/u/vineeku6/storage/train-data/ellipsis/raw-5/valid_src.txt'
GOLD='/u/vineeku6/storage/train-data/ellipsis/raw-5/valid_target.txt'
OUT=out-$MODEL.txt

python translate_all.py $MODEL.npz $INPUT $OUT --all
#python translate_all.py $MODEL.npz $INPUT $OUT

python convert_symbols.py $INPUT $OUT
./multi-bleu.perl $GOLD.nounk < $OUT.nounk &> bleu-$MODEL.out

python report.py $INPUT $OUT $GOLD $MODEL 
