from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
from providedcode.dependencygraph import DependencyGraph
import sys

tp = TransitionParser.load('english.model')
sentences = []
for sentence in sys.stdin:
    sentence = sentence.strip()
    sentence = DependencyGraph.from_sentence(sentence)
    sentences.append(sentence)

parsed = tp.parse(sentences)
for parse in parsed:
    print parse.to_conll(10).encode('utf-8')
