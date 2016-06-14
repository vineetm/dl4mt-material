#!/bin/sh
UNK_20='/u/vineeku6/data-el/fixed-v3/stopw'
WORD_VEC_300='/u/vineeku6/storage/word-embeddings/trained/wiki-300d-ep4/wiki.vectors'
WORD_VEC_200='/u/vineeku6/storage/word-embeddings/trained/wiki-200d/wiki.vectors'
WORD_VEC_100='/u/vineeku6/storage/word-embeddings/trained/wiki-100d/wiki.vectors'
WORD_VEC_50='/u/vineeku6/storage/word-embeddings/trained/wiki-50d/wiki.vectors'

QUEUE='x86_short'
#QUEUE='x86_excl'

BASE=$UNK_20
VOCAB=242

##Experiments with d=200
WORD_VEC=$WORD_VEC_200
D=200

H=100
MODEL='ellipsis-exp40'
ALPHA=1.0
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp41'
ALPHA=10.0
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

##Experiments with d=100
WORD_VEC=$WORD_VEC_100
D=100

H=100
MODEL='ellipsis-exp42'
ALPHA=1.0
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp43'
ALPHA=10.0
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

##Experiments with d=50
WORD_VEC=$WORD_VEC_50
D=50

H=50
MODEL='ellipsis-exp44'
ALPHA=1.0
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp45'
ALPHA=10.0
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA
