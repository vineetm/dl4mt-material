#!/bin/sh
UNK_20='/u/vineeku6/data-el/fixed-v3/stopw'
WORD_VEC_300='/u/vineeku6/storage/word-embeddings/trained/wiki-300d-ep4/wiki.vectors'
WORD_VEC_200='/u/vineeku6/storage/word-embeddings/trained/wiki-200d/wiki.vectors'
WORD_VEC_100='/u/vineeku6/storage/word-embeddings/trained/wiki-100d/wiki.vectors'
WORD_VEC_50='/u/vineeku6/storage/word-embeddings/trained/wiki-50d/wiki.vectors'

QUEUE='x86_short'
#QUEUE='x86_excl'

BASE=$UNK_20
VOCAB=825

##Experiments with d=50
WORD_VEC=$WORD_VEC_200
D=200

H=200
MODEL='ellipsis-exp30'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp31'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp32'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp33'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp34'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

H=100
MODEL='ellipsis-exp35'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp36'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp37'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp38'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp39'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA
