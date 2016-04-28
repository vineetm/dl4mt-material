#!/bin/sh

MODEL=$1
SUFFIX=$2
INPUT=$3
GOLD=$4

jbsub -name report-$SUFFIX -out report-$SUFFIX.out -mem 10g -cores 2+1 -queue x86_short  python generate_report.py $MODEL $INPUT $GOLD --suffix $SUFFIX