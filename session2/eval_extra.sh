#!/bin/sh
UNK_20='/u/vineeku6/data-el/fixed-v2/unk-seq'

BASE=$UNK_20
./submit-eval.sh ellipsis-exp30 $BASE

./submit-eval.sh ellipsis-exp31 $BASE

./submit-eval.sh ellipsis-exp32 $BASE

./submit-eval.sh ellipsis-exp33 $BASE

./submit-eval.sh ellipsis-exp34 $BASE

./submit-eval.sh ellipsis-exp35 $BASE

./submit-eval.sh ellipsis-exp36 $BASE

./submit-eval.sh ellipsis-exp37 $BASE

./submit-eval.sh ellipsis-exp38 $BASE

./submit-eval.sh ellipsis-exp39 $BASE
