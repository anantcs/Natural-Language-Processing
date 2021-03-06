import sys
import nltk
import math
import time
import itertools
from collections import defaultdict

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for line in brown_train:
        brown_wordline = []
        brown_tagline = []
        brown_wordline.append(START_SYMBOL)
        brown_wordline.append(START_SYMBOL)
        brown_tagline.append(START_SYMBOL)
        brown_tagline.append(START_SYMBOL)
        words = line.split()
        #del words[-1]
        for word in words:
            index = word.rfind('/')
            tempword = word[0:index]
            temptag = word[index+1:]
            brown_wordline.append(tempword)
            brown_tagline.append(temptag)
        brown_wordline.append(STOP_SYMBOL)
        brown_tagline.append(STOP_SYMBOL)
        brown_words.append(brown_wordline)
        brown_tags.append(brown_tagline)
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}

    
    total_tags=0
    unigram_p = {}
    for line in brown_tags:
        total_tags += len(line)
    unigram_count = defaultdict(int)
    for line in brown_tags:
        tags = line
        for tag in tags:
            unigram_count[tag] += 1
    for unigram in unigram_count:
        key = (unigram,)
        unigram_p[key] = math.log(unigram_count[unigram]/float(total_tags), 2)
    
    bigram_p = {}
    bigram_count = defaultdict(int)
    for line in brown_tags:
        tags = line
        bigram_tuples = list(nltk.bigrams(tags))
        for tag in bigram_tuples:
            bigram_count[tag] += 1
    for bigram in bigram_count:
        key = bigram
        if unigram_count[bigram[0]] == 0:
            unigram_count[bigram[0]] = len(brown_tags)
            unigram_p[(bigram[0],)] = math.log(unigram_count[bigram[0]]/float(total_tags), 2)
            bigram_p[key] = math.log(bigram_count[bigram]/float(unigram_count[bigram[0]]), 2)
        else:
            bigram_p[key] = math.log(bigram_count[bigram]/float(unigram_count[bigram[0]]), 2)
    
    trigram_p = {}
    trigram_count = defaultdict(int)
    for line in brown_tags:
        tags = line
        trigram_tuples = list(nltk.trigrams(tags))
        for tag in trigram_tuples:
            trigram_count[tag] += 1
    for trigram in trigram_count:
        key = trigram
        #print trigram[0:2], type(trigram[0:2]), bigram_count[trigram[0:2]]
        if bigram_count[trigram[0:2]] == 0:
            bigram_count[trigram[0:2]] = len(brown_tags)
            bigram_p[trigram[0:2]] = math.log(bigram_count[trigram[0:2]]/float(unigram_count[bigram[0]]),2)
            trigram_p[key] = math.log(trigram_count[trigram]/float(bigram_count[trigram[0:2]]),2)
        else:
            trigram_p[key] = math.log(trigram_count[trigram]/float(bigram_count[trigram[0:2]]), 2)

    q_values = trigram_p

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    wordcount = defaultdict(int)
    for words in brown_words:
        for word in words:
            wordcount[word] += 1
    for word in wordcount:
        if wordcount[word] > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for words in brown_words:
        brown_templine = []
        for word in words:
            if word not in known_words:
                brown_templine.append(RARE_SYMBOL)
            else:
                brown_templine.append(word)
        brown_words_rare.append(brown_templine)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    #creating a count of tags
    tagcount = defaultdict(int)
    for tags in brown_tags:
        for tag in tags:
            tagcount[tag] += 1
    #creating a count of (word, tag) tuples
    tuplecount = defaultdict(int)
    #print len(brown_words_rare), len(brown_tags)
    for words, tags in zip(brown_words_rare, brown_tags):
        for word, tag in zip(words, tags):
            tuplecount[tuple((word, tag))] += 1
    e_values = {}
    for tup in tuplecount:
        e_values[tup] = math.log(tuplecount[tup]/float(tagcount[tup[1]]),2)
    #creating a set of distinct tags
    taglist = set([])
    for tags in brown_tags:
        for tag in tags:
            if tag not in taglist:
                taglist.add(tag)
            else:
                pass
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

def forward(brown_dev_words,taglist, known_words, q_values, e_values):
    probs = []
    tagged = []
    pi = defaultdict(float)
    backpointers = {}
    backpointers[(-1, START_SYMBOL, START_SYMBOL)] = START_SYMBOL
    pi[(-1, START_SYMBOL, START_SYMBOL)] = 0.0

    taglist_s = set(taglist)
    taglist_s.remove(START_SYMBOL)
    taglist_s.remove(STOP_SYMBOL)
    taglist = list(taglist_s)

    known_words_set = set(known_words)

    for line in brown_dev_words:
        tokens = []
        for word in line:
            if word in known_words:
                tokens.append(word)
            else:
                tokens.append(RARE_SYMBOL)

        k = 0
        for w in taglist:
            word_tag = (tokens[k], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            state = tuple((k, START_SYMBOL, w))
            pi[state] = pi[(-1, START_SYMBOL, START_SYMBOL)]
            if trigram not in q_values:
                pi[state] += LOG_PROB_OF_ZERO
            else:
                pi[state] += q_values[trigram]
            if word_tag not in e_values:
                pi[state] += LOG_PROB_OF_ZERO
            else:
                pi[state] += e_values[word_tag]
            backpointers[state] = START_SYMBOL

        k = 1
        for w in taglist:
            for u in taglist:
                word_tag = (tokens[k], u)
                trigram = (START_SYMBOL, w, u)
                state = tuple((k,w,u))
                pi[state] = pi.get((k-1, START_SYMBOL, w), LOG_PROB_OF_ZERO)
                if trigram not in q_values:
                    pi[state] += LOG_PROB_OF_ZERO
                else:
                    pi[state] += q_values[trigram]
                if word_tag not in e_values:
                    pi[state] += LOG_PROB_OF_ZERO
                else:
                    pi[state] += e_values[word_tag]
                backpointers[state] = START_SYMBOL

        for k in range(2, len(tokens)):
            for u in taglist:
                for v in taglist:
                    max_probability = float('-Inf')
                    max_tag = ''
                    for w in taglist:
                        score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + e_values.get((tokens[k], v), LOG_PROB_OF_ZERO)
                        if (score > max_probability):
                            max_probability,max_tag = score, w
                    backpointers[(k, u, v)] = max_tag
                    pi[(k, u, v)] = max_probability

        max_v, max_u = 0, 0 
        max_probability = float('-Inf')
    
        for (u, v) in itertools.product(taglist, taglist):
            value = pi.get((len(line) - 1, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if value > max_probability:
                max_probability, max_u, max_v = value, u, v
        #print max_probability
        probs.append(max_probability)
        tags = []
        tags.extend((max_v, max_u))

        for count, k in enumerate(range(len(line) - 3, -1, -1)):
            tags.append(backpointers[(k + 2, tags[count + 1], tags[count])])

        tagged_sentence = []
        tags.reverse()
        for k in range(0, len(line)):
            tagged_sentence += [line[k], "/", str(tags[k]), " "]
        tagged_sentence.append('\n')
        tagged.append(''.join(tagged_sentence))
    return probs

# This function takes the output of forward() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(str(sentence) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    pi = defaultdict(float)
    backpointers = {}
    backpointers[(-1, START_SYMBOL, START_SYMBOL)] = START_SYMBOL
    pi[(-1, START_SYMBOL, START_SYMBOL)] = 0.0

    taglist_s = set(taglist)
    taglist_s.remove(START_SYMBOL)
    taglist_s.remove(STOP_SYMBOL)
    taglist = list(taglist_s)

    known_words_set = set(known_words)

    for line in brown_dev_words:
        tokens = []
        for word in line:
            if word in known_words:
                tokens.append(word)
            else:
                tokens.append(RARE_SYMBOL)

        k = 0
        for w in taglist:
            word_tag = (tokens[k], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            state = tuple((k, START_SYMBOL, w))
            pi[state] = pi[(-1, START_SYMBOL, START_SYMBOL)]
            if trigram not in q_values:
                pi[state] += LOG_PROB_OF_ZERO
            else:
                pi[state] += q_values[trigram]
            if word_tag not in e_values:
                pi[state] += LOG_PROB_OF_ZERO
            else:
                pi[state] += e_values[word_tag]
            backpointers[state] = START_SYMBOL

        k = 1
        for w in taglist:
            for u in taglist:
                word_tag = (tokens[k], u)
                trigram = (START_SYMBOL, w, u)
                state = tuple((k,w,u))
                pi[state] = pi.get((k-1, START_SYMBOL, w), LOG_PROB_OF_ZERO)
                if trigram not in q_values:
                    pi[state] += LOG_PROB_OF_ZERO
                else:
                    pi[state] += q_values[trigram]
                if word_tag not in e_values:
                    pi[state] += LOG_PROB_OF_ZERO
                else:
                    pi[state] += e_values[word_tag]
                backpointers[state] = START_SYMBOL

        for k in range(2, len(tokens)):
            for u in taglist:
                for v in taglist:
                    max_probability = float('-Inf')
                    max_tag = ''
                    for w in taglist:
                        score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + e_values.get((tokens[k], v), LOG_PROB_OF_ZERO)
                        if (score > max_probability):
                            max_probability,max_tag = score, w
                    backpointers[(k, u, v)] = max_tag
                    pi[(k, u, v)] = max_probability

        max_v, max_u = 0, 0 
        max_probability = float('-Inf')
    
        for (u, v) in itertools.product(taglist, taglist):
            value = pi.get((len(line) - 1, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if value > max_probability:
                max_probability, max_u, max_v = value, u, v
        #print max_probability
        tags = []
        tags.extend((max_v, max_u))

        for count, k in enumerate(range(len(line) - 3, -1, -1)):
            tags.append(backpointers[(k + 2, tags[count + 1], tags[count])])

        tagged_sentence = []
        tags.reverse()
        for k in range(0, len(line)):
            tagged_sentence += [line[k], "/", str(tags[k]), " "]
        tagged_sentence.append('\n')
        tagged.append(''.join(tagged_sentence))

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    #training = []
    #for i in xrange(len(brown_words)):
    #    temp_training = []
    #    for j in xrange(len(brown_words[i])):
    #        temp_training.append(tuple((unicode(brown_words[i][j]), unicode(brown_tags[i][j]))))
    #    training.append(temp_training)
    #for train in training:
    #    print type(train), type(train[0])
    default_tagger = nltk.DefaultTagger("NOUN")
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    for words in brown_dev_words:        
        temp_tagged = []
        tagged_sentence = trigram_tagger.tag(words)
        #print tagged_sentence
        for word in tagged_sentence:
            temp_tagged.append(str(word[0]+'/'+str(word[1])))
        temp_sentence = " ".join(temp_tagged)
        temp_sentence = temp_sentence + "\r\n"
        tagged.append(temp_sentence)
    #for tag in tagged:
    #    print tag
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q7_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare



    # open Brown development data (question 6)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # question 5
    forward_probs = forward(brown_dev_words,taglist, known_words, q_values, e_values)
    q5_output(forward_probs, OUTPUT_PATH + 'B5.txt')

    # do viterbi on brown_dev_words (question 6)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 6 output
    q6_output(viterbi_tagged, OUTPUT_PATH + 'B6.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 7 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B7.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
