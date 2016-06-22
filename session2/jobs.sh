#!/bin/sh
DATA='/u/vineeku6/data-el/entity/freq-20'
WORD_VEC_300='/u/vineeku6/storage/word-embeddings/trained/wiki-300d-ep4/wiki.vectors'
WORD_VEC_200='/u/vineeku6/storage/word-embeddings/trained/wiki-200d/wiki.vectors'
WORD_VEC_100='/u/vineeku6/storage/word-embeddings/trained/wiki-100d/wiki.vectors'
WORD_VEC_50='/u/vineeku6/storage/word-embeddings/trained/wiki-50d/wiki.vectors'

QUEUE='x86_short'
#QUEUE='x86_excl'

BASE=$DATA
VOCAB=742

##Experiments with d=50
WORD_VEC=$WORD_VEC_50
D=50

H=50
MODEL='ellipsis-exp00'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp01'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp02'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp03'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp04'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

H=40
MODEL='ellipsis-exp05'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp06'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp07'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp08'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp09'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

H=30
MODEL='ellipsis-exp10'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp11'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp12'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp13'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp14'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

##Experiments with d=100
WORD_VEC=$WORD_VEC_100
D=100

H=100
MODEL='ellipsis-exp15'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp16'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp17'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp18'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp19'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

##Experiments with d=200
WORD_VEC=$WORD_VEC_200
D=200

H=200
MODEL='ellipsis-exp20'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp21'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp22'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp23'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp24'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

H=100
MODEL='ellipsis-exp25'
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D

MODEL='ellipsis-exp26'
ALPHA=0.1
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp27'
ALPHA=0.01
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp28'
ALPHA=0.001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

MODEL='ellipsis-exp29'
ALPHA=0.0001
jbsub -name $MODEL -out $MODEL.out -mem 10g -queue $QUEUE -cores 2+1 python train.py $BASE $WORD_VEC $MODEL --srcwords $VOCAB --targetwords $VOCAB --dimhidden $H --dimword $D --alphac $ALPHA

##Experiments with d=300
WORD_VEC=$WORD_VEC_200
D=300

H=300
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

H=200
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
