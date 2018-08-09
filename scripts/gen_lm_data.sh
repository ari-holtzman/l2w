#!/bin/bash
for input in disc_train.txt valid.txt test.txt
do
    python generate.py --cuda --data "$1$input.context" --lm "$2" --dic "$3" --gen_disc_data --beam_size 4 --out "$1$input.generated_continuation"
done
