#!/bin/sh

MODEL=$1
SUFFIX=$2
INPUT='/u/vineeku6/data/question-generation/evaluation/answers.txt'

jbsub -name report-$SUFFIX -out report-$SUFFIX.out -mem 10g -cores 2+1 -queue x86_short  python generate_report.py $MODEL --suffix $SUFFIX $INPUT
