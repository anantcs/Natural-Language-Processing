NLP COMS 4705
Assignment 3
Anant Sharma
as5194

1. a) b)
Implemented

2. a)
Verbs could have been stemmed
Fractions, symbols haven't taken into account, could have been

b) Dict may have been better because np matrix takes too long to iterate.
I had to make sure that all the backpointers were safely maintained to avoid
any errors.


2.c)

python part2.py data/english/train.conll data/english/dev.conll output model

UAS: 65.57
LAS: 54.91

python part2.py data/english/train.conll data/english/dev.conll output
model_pos --pos_d 25

Training loss: (-84.68366265296936, -37.89213800430298)
UAS: 79.12
LAS: 74.57

python part2.py data/korean/train.conll data/korean/dev.conll output
model_korean --pos_d 25

Training loss: (-60.79183542728424, -26.591010332107544)
UAS: 73.37
LAS: 58.36

python part2.py data/swedish/train.conll data/swedish/dev.conll output
model_swedish --pos_d 25

Training loss: (-1915.8408843278885, -59.673001646995544)
UAS: 83.29
LAS: 72.72

