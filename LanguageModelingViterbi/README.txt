COMS 4705 - Natural Language Processing - Spring 2017
Assignment 1
Language Modeling and Part of Speech Tagging

PART A

1)
UNIGRAM near -12.4560686964
BIGRAM near the -1.56187888761
TRIGRAM near the ecliptic -5.39231742278

2)
UNIGRAM - The perplexity is 1052.4865859
BIGRAM - The perplexity is 53.8984761198
TRIGRAM - The perplexity is 5.7106793082

3)
The perplexity is 12.953323319

4)
I was expecting the linear interpolation model to give a better result because
it weights all the three models - unigram, bigram and trigram. But it might be
that we haven't chosen the weights of the three lambdas well. If we had chose
them well, the perplexity might have been better for the linear interpolation
model.

5)
Sample1.txt - The perplexity is 9.33239078711
Sample2.txt - The perplexity is 21890298275.0

A language model is better when the perplexity of the model is lower. By this
standard, the first sample belongs to the Brown dataset, and the second doesn't
belong to the Brown dataset.

PART B


2)
TRIGRAM CONJ ADV NOUN -4.46650366731
TRIGRAM DET NUM NOUN -0.713200128516
TRIGRAM NOUN PRT CONJ -6.38503274104

4)
* * 0.0
midnight NOUN -13.1814628813
Place VERB -15.4538814891
primary ADJ -10.0668014957
STOP STOP 0.0
_RARE_ VERB -3.17732085089
_RARE_ X -0.546359661497

5)
-201.335182531

6)
Percent correct tags: 93.3226493638

7)
Percent correct tags: 87.9985146677

RUNTIME

Part A time: 9.551093 sec 
Part B time: 751.973234 sec
Total Time: ~760 secs
