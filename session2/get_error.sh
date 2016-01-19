#!/bin/sh
MODEL=$1
SUFFIX=$2
SRC='/u/vineeku6/data/question-generation/yahoo-answers/valid_a.txt'
TGT='/u/vineeku6/data/question-generation/yahoo-answers/valid_q.txt'
jbsub -name test-$SUFFIX -out test-$SUFFIX.out -cores 2+1 -mem 10g -queue x86_short python get_error.py $MODEL $SRC $TGT
