import math
import nltk
import time
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    total_words=0
    unigram_p = {}
    for line in training_corpus:
        total_words += len(line.split(" "))
    unigram_count = defaultdict(int)
    for line in training_corpus:
        words = line.split(" ")
        #words.insert(0, START_SYMBOL)
        words.insert(-1,STOP_SYMBOL)
        del words[-1]
        for word in words:
            unigram_count[word] += 1
    for unigram in unigram_count:
        key = (unigram,)
        unigram_p[key] = math.log(unigram_count[unigram]/float(total_words), 2)
    
    bigram_p = {}
    bigram_count = defaultdict(int)
    for line in training_corpus:
        words = line.split(" ")
        words.insert(0, START_SYMBOL)
        words.insert(-1, STOP_SYMBOL)
        del words[-1]
        bigram_tuples = list(nltk.bigrams(words))
        for tup in bigram_tuples:
            bigram_count[tup] += 1
    for bigram in bigram_count:
        key = bigram
        #print key, bigram[0]
        #print unigram_count[bigram[0]]
        if unigram_count[bigram[0]] == 0:
            unigram_count[bigram[0]] = len(training_corpus)
            unigram_p[(bigram[0],)] = math.log(unigram_count[bigram[0]]/float(total_words), 2)
            bigram_p[key] = math.log(bigram_count[bigram]/float(unigram_count[bigram[0]]), 2)
        else:
            bigram_p[key] = math.log(bigram_count[bigram]/float(unigram_count[bigram[0]]), 2)
    
    trigram_p = {}
    trigram_count = defaultdict(int)
    for line in training_corpus:
        words = line.split(" ")
        words.insert(0, START_SYMBOL)
        words.insert(0, START_SYMBOL)
        words.insert(-1, STOP_SYMBOL)
        del words[-1]
        trigram_tuples = list(nltk.trigrams(words))
        for tup in trigram_tuples:
            trigram_count[tup] += 1
    for trigram in trigram_count:
        key = trigram
        #print trigram[0:2], type(trigram[0:2]), bigram_count[trigram[0:2]]
        if bigram_count[trigram[0:2]] == 0:
            bigram_count[trigram[0:2]] = len(training_corpus)
            bigram_p[trigram[0:2]] = math.log(bigram_count[trigram[0:2]]/float(unigram_count[bigram[0]]),2)
            trigram_p[key] = math.log(trigram_count[trigram]/float(bigram_count[trigram[0:2]]),2)
        else:
            trigram_p[key] = math.log(trigram_count[trigram]/float(bigram_count[trigram[0:2]]), 2)
    
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for line in corpus:
        score = 0
        words = line.split(" ")
        words.insert(-1, STOP_SYMBOL)
        del words[-1]
        #print words
        if n == 1:
            for i in range(len(words)):
                #print i, words[i], (words[i],)
                #print ngram_p[(words[i],)], score
                score += ngram_p[(words[i],)]
        elif n == 2:
            words.insert(0, START_SYMBOL)
            bigram_tuples = list(nltk.bigrams(words))
            #print bigram_tuples
            #print len(bigram_tuples)
            for i in range(len(bigram_tuples)):
                score += ngram_p[bigram_tuples[i]]
        elif n == 3:
            words.insert(0, START_SYMBOL)
            words.insert(0, START_SYMBOL)
            trigram_tuples = list(nltk.trigrams(words))
            #print len(trigram_tuples)
            for i in range(len(trigram_tuples)):
                score += ngram_p[trigram_tuples[i]]
                #print i, trigram_tuples[i], ngram_p[trigram_tuples[i]], score
        else:
            pass
        scores.append(score)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

#TODO: IMPLEMENT THIS FUNCTION    
# Calculcates the perplexity of a language model
# scores_file: one of the A2 output files of scores 
# sentences_file: the file of sentences that were scores (in this case: data/Brown_train.txt) 
# This function returns a float, perplexity, the total perplexity of the corpus
def calc_perplexity(scores_file, sentences_file):

    perplexity = 0
    infile = open(scores_file, 'r')
    scores = infile.readlines()
    infile.close()
    infile = open(sentences_file, 'r')
    sentences = infile.readlines()
    infile.close()

    total_scores = 0
    total_words = 0
    #print len(scores), len(sentences)

    for i in range(len(scores)):
        total_scores += float(scores[i].split()[0])
        total_words += len(sentences[i].split(" "))
    total_scores *= -1
    print total_scores, total_words
    perplexity = 2**(float(total_scores)/total_words)

    return perplexity 

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for line in corpus:
        line_probability = 1
        unigram_score, bigram_score, trigram_score = 0,0,0
        unigram_present, bigram_present, trigram_present = True, True, True
        words = line.split(" ")
        words.insert(0, START_SYMBOL)
        words.insert(0, START_SYMBOL)
        words.insert(-1, STOP_SYMBOL)
        del words[-1]
        trigram_tuples = list(nltk.trigrams(words))
        for tup in trigram_tuples:
            if tup[2] in unigrams:
                unigram_score = math.pow(2, unigrams[(tup[2],)])
            else:
                unigram_present = False
            if tup[1:] in bigrams:
                bigram_score = math.pow(2, bigrams[tup[1:]])
            else:
                bigram_present = False
            if tup in trigrams:
                trigram_score = math.pow(2, trigrams[tup])
            else:
                trigram_present = False
        #print unigram_score, bigram_score, trigram_score
            line_probability *= ( ( unigram_score + bigram_score + trigram_score ) / 3)
        if unigram_present == False and bigram_present == False and trigram_present == False:
            score = MINUS_INFINITY_SENTENCE_LOG_PROB
        elif line_probability == 0:
            score = MINUS_INFINITY_SENTENCE_LOG_PROB
        else:
            score = math.log(line_probability, 2)
        print score
        scores.append(score)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
