#!/bin/sh
MODEL=$1
INPUT=$2/valid_src.txt
GOLD=$2/valid_target.txt
OUT=out-$MODEL.txt

python translate_all.py $MODEL.npz $INPUT $OUT --all

python convert_symbols.py $INPUT $OUT
./multi-bleu.perl $GOLD.nounk < $OUT.nounk &> bleu-$MODEL.out

python report.py $INPUT $OUT $GOLD $MODEL 
