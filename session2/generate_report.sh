#!/bin/sh

SUFFIX='yahoo-reg2'
INPUT=answers.txt
MODEL='best-yahoo-reg2.npz'
SRC_DICT='/u/vineeku6/data/question-generation/yahoo-answers/a.txt.pkl'
TGT_DICT='/u/vineeku6/data/question-generation/yahoo-answers/q.txt.pkl'

jbsub -name report-$SUFFIX -out report-$SUFFIX.out -mem 10g python generate_report.py $MODEL $SRC_DICT --tgtd $TGT_DICT --suffix $SUFFIX $INPUT

