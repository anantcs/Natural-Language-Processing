from dynet import *
import argparse
from utils import Corpus
import random
import numpy as np
from bleu import get_bleu_score
import json

RNN_BUILDER = GRUBuilder
EOS = "</s>"

class nmt_dynet:

    def __init__(self, src_vocab_size, tgt_vocab_size, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word, word_d, gru_d, gru_layers):

        # initialize variables
        self.gru_layers = gru_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_word2idx = src_word2idx
        self.src_idx2word = src_idx2word
        self.tgt_word2idx = tgt_word2idx
        self.tgt_idx2word = tgt_idx2word
        self.word_d = word_d
        self.gru_d = gru_d

        self.model = Model()

        # the embedding paramaters
        self.source_embeddings = self.model.add_lookup_parameters((self.src_vocab_size, self.word_d))
        self.target_embeddings = self.model.add_lookup_parameters((self.tgt_vocab_size, self.word_d))


        # YOUR IMPLEMENTATION GOES HERE
        # project the decoder output to a vector of tgt_vocab_size length
        self.output_w = self.model.add_parameters((self.tgt_vocab_size,self.gru_d))
        self.output_b = self.model.add_parameters((self.tgt_vocab_size,))

        # encoder network
        # the foreword rnn
        self.fwd_RNN = RNN_BUILDER(self.gru_layers, self.word_d, self.gru_d,
        self.model)
        # the backword rnn
        self.bwd_RNN = RNN_BUILDER(self.gru_layers, self.word_d, self.gru_d,
        self.model)

        # decoder network
        self.dec_RNN = RNN_BUILDER(self.gru_layers, self.gru_d*2 + self.word_d,
        self.gru_d, self.model)


    def embed_src_sentence(self, src_sentence):
        src_sentence = [self.src_word2idx[word] for word in src_sentence]
        embedded_src_sentence = [self.source_embeddings[idx] for idx in src_sentence]
        return embedded_src_sentence

    def embed_tgt_sentence(self, tgt_sentence):
        tgt_sentence = [self.tgt_word2idx[word] for word in tgt_sentence]
        embedded_tgt_sentence = [self.target_embeddings[idx] for idx in tgt_sentence]
        return embedded_tgt_sentence
    
    def encode(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return encoding of the source sentence
        '''
        # YOUR IMPLEMENTATION GOES HERE
        input_sentence = self.embed_src_sentence(src_sentence)
        fwd_encode = self.fwd_RNN.initial_state()
        bkd_encode = self.bwd_RNN.initial_state()
        fwd_vectors = []
        bkd_vectors = []
        
        for i in range(0, len(src_sentence), 1):
            fwd_encode = fwd_encode.add_input(src_sentence[i])
            fwd_vectors.append(fwd_encode.output())
        
        for i in range(len(src_sentence)-1, -1, -1):
            bkd_encode = bkd_encode.add_input(src_sentence[i])
            bkd_vectors.append(bkd_encode.output())
        
        encoded_sentence = [concatenate(list(p)) for p in zip(fwd_vectors, bkd_vectors)]
        return encoded_sentence

    def decode(self, encoded_sentence, tgt_sentence):
        #embedded_tgt_sentence = self.embed_tgt_sentence(tgt_sentence)
        input_mat = encoded_sentence[-1]
        
        last_tgt_embeddings = zeroes((self.word_d, 1)) 
        s = self.dec_RNN.initial_state()
        loss = []

        for word in tgt_sentence:
            w = parameter(self.output_w)
            b = parameter(self.output_b)
            vector = concatenate([input_mat, last_tgt_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = softmax(out_vector)
            last_tgt_embeddings = self.target_embeddings[self.tgt_word2idx[word]]
            loss.append(-log(pick(probs,self.tgt_word2idx[word]))) 
        return esum(loss)

    def get_loss(self, src_sentence, tgt_sentence):
        '''
        src_sentence: words in src sentence
        tgt_sentence: words in tgt sentence
        return loss for this source target sentence pair
        '''
        renew_cg()
        # YOUR IMPLEMENTATION GOES HERE
        embedded_src_sentence = self.embed_src_sentence(src_sentence)
        encoded_sentence = self.encode(embedded_src_sentence)
        return self.decode(encoded_sentence, tgt_sentence)


    def generate(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return list of words in the target sentence
        '''
        renew_cg()

        # YOUR IMPLEMENTATION GOES HERE
        embedded_src_sentence = self.embed_src_sentence(src_sentence)
        input_mat = self.encode(embedded_src_sentence)[-1]

        last_tgt_embeddings = zeroes((self.word_d, 1)) 
        s = self.dec_RNN.initial_state()

        out = []
        nEOS = 0
        while(True):
            w = parameter(self.output_w)
            b = parameter(self.output_b)
            vector = concatenate([input_mat, last_tgt_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_tgt_embeddings = self.target_embeddings[next_word]
            out.append(self.tgt_idx2word[next_word])
            if self.tgt_idx2word[next_word] == EOS:
                break
            if nEOS > 10 * len(src_sentence):
                break
            nEOS += 1
        return out

    def translate_all(self, src_sentences):
        translated_sentences = []
        for src_sentence in src_sentences:
            # print src_sentence
            translated_sentences.append(self.generate(src_sentence))

        return translated_sentences

    # save the model, and optionally the word embeddings
    def save(self, filename):

        self.model.save(filename)
        embs = {}
        if self.src_idx2word:
            src_embs = {}
            for i in range(self.src_vocab_size):
                src_embs[self.src_idx2word[i]] = self.source_embeddings[i].value()
            embs['src'] = src_embs

        if self.tgt_idx2word:
            tgt_embs = {}
            for i in range(self.tgt_vocab_size):
                tgt_embs[self.tgt_idx2word[i]] = self.target_embeddings[i].value()
            embs['tgt'] = tgt_embs

        if len(embs):
            with open(filename + '_embeddings.json', 'w') as f:
                json.dump(embs, f)

def get_val_set_loss(network, val_set):
        loss = []
        for src_sentence, tgt_sentence in zip(val_set.source_sentences, val_set.target_sentences):
            loss.append(network.get_loss(src_sentence, tgt_sentence).value())

        return sum(loss)

def main(train_src_file, train_tgt_file, dev_src_file, dev_tgt_file, model_file, num_epochs, embeddings_init = None, seed = 0):
    print('reading train corpus ...')
    train_set = Corpus(train_src_file, train_tgt_file)
    # assert()
    print('reading dev corpus ...')
    dev_set = Corpus(dev_src_file, dev_tgt_file)

    print 'Initializing simple neural machine translator:'
    # src_vocab_size, tgt_vocab_size, tgt_idx2word, word_d, gru_d, gru_layers
    encoder_decoder = nmt_dynet(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx, train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, 50, 50, 2)

    trainer = SimpleSGDTrainer(encoder_decoder.model)

    sample_output = np.random.choice(len(dev_set.target_sentences), 5, False)
    losses = []
    best_bleu_score = 0
    for epoch in range(num_epochs):
        print 'Starting epoch', epoch
        # shuffle the training data
        combined = list(zip(train_set.source_sentences, train_set.target_sentences))
        random.shuffle(combined)
        train_set.source_sentences[:], train_set.target_sentences[:] = zip(*combined)

        print 'Training . . .'
        sentences_processed = 0
        for src_sentence, tgt_sentence in zip(train_set.source_sentences, train_set.target_sentences):
            loss = encoder_decoder.get_loss(src_sentence, tgt_sentence)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            sentences_processed += 1
            if sentences_processed % 4000 == 0:
                print 'sentences processed: ', sentences_processed

        # Accumulate average losses over training to plot
        val_loss = get_val_set_loss(encoder_decoder, dev_set)
        print 'Validation loss this epoch', val_loss
        losses.append(val_loss)

        print 'Translating . . .'
        translated_sentences = encoder_decoder.translate_all(dev_set.source_sentences)

        print('translating {} source sentences...'.format(len(sample_output)))
        for sample in sample_output:
            print('Target: {}\nTranslation: {}\n'.format(' '.join(dev_set.target_sentences[sample]),
                                                                         ' '.join(translated_sentences[sample])))

        bleu_score = get_bleu_score(translated_sentences, dev_set.target_sentences)
        print 'bleu score: ', bleu_score
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            # save the model
            encoder_decoder.save(model_file)

    print 'best bleu score: ', best_bleu_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
#     parser.add_argument('model_type')
    parser.add_argument('train_src_file')
    parser.add_argument('train_tgt_file')
    parser.add_argument('dev_src_file')
    parser.add_argument('dev_tgt_file')
    parser.add_argument('model_file')
    parser.add_argument('--num_epochs', default = 20, type = int)
    parser.add_argument('--embeddings_init')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--dynet-mem')

    args = vars(parser.parse_args())
    args.pop('dynet_mem')

    main(**args)
