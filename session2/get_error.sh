#!/bin/sh
MODEL=$1
SUFFIX=$2
SRC='/u/vineeku6/storage/train-data/ellipsis/exp7/valid_src.txt'
TGT='/u/vineeku6/storage/train-data/ellipsis/exp7/valid_target.txt'
jbsub -name test-$SUFFIX -out test-$SUFFIX.out -cores 2+1 -mem 10g -queue x86_short python get_error.py $MODEL $SRC $TGT
