#!/bin/sh

SUFFIX='chk'
INPUT='/u/vineeku6/data/question-generation/evaluation/answers.txt'
MODEL='test.npz'
SRC_DICT='/u/vineeku6/data/question-generation/yahoo-answers/data.txt.pkl'
#TGT_DICT='/u/vineeku6/data/question-generation/yahoo-answers/q.txt.pkl'

jbsub -name report-$SUFFIX -out report-$SUFFIX.out -mem 10g -cores 2+1 -queue x86_short  python generate_report.py $MODEL $SRC_DICT --suffix $SUFFIX $INPUT

