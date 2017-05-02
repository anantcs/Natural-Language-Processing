
DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def calc_probabilities(training_corpus):
    total_words = 0
    for line in training_corpus:
      total_words += len(line.split())
    unigram_p = {}
    count = defaultdict(int)
    for line in data:
      line += STOP_SYMBOL
      for word in line.split():
        count[word] += 1
    for c in count_keys:
      unigram_p[c] = math.log(count[c]/float(total_words),2)
    bigram_p = {}
    trigram_p = {}
    return unigram_p, bigram_p, trigram_p

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()
    print corpus

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)
    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    i = 0
    for unigram in unigrams_keys:
        print 'coming here'
        if i == 10:
	    break
        print unigram[0] + ' ' + str(unigrams[unigram])
        i = i + 1
        #outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
