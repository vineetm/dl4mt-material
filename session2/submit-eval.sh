#!/bin/sh

MODEL=$1
jbsub -name report-$MODEL -out report-$MODEL.out -mem 10g -cores 2+1 -queue x86_short ./evaluate.sh $MODEL $2
#jbsub -name report-$MODEL -out report-$MODEL.out -mem 10g -cores 2+1 -queue x86_excl ./evaluate.sh $MODEL $2
