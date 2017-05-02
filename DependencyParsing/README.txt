COMS 4705 NLP
ASSIGNMENT TWO
Anant Sharma
as5194

1. a) The images have been generated and named accordingly.

b) To check if a dependency graph is projective, the code is first of all
adding all the arc in a set. Then for every arc, the code checks if there's an
arc that's either starting outside the parent and the child and ending between
them, or starting between the parent and the child and ending outside them.

2. a) Implemented

b)
UAS: 0.279305354559 
LAS: 0.178726483357
UAS is the #correct heads identified which is 27.9%, where LAS is the #number
of correct arcs(both head and dependent) identified, which is 17.87% in this
case.
The badfeatures model consisted of the following features - stack top form,
stack top feats, buffer top form, and buffer top feats, along with
leftmostdependent and rightmostdependent.

3. a) Implemented

b)

c) Implemented

d) English.model
UAS: 0.7902200489
LAS: 0.753545232274

Swedish.model
UAS: 0.802460202605 
LAS: 0.717076700434

FEATURES ADDED

form, postag, lemma - are the three features that I've added for the elements
at the top of the stack and the buffer
form, postag - second topmost element of the stack
form, postag - second element of the buffer
postag - third and fourth element of the buffer

OBSERVATIONS:
Adding these features dramatically increases the LAS values for both the
Swedish and English models. For Swedish, the LAS value increases from 17.8% to
71%, which shows that the features that we added were indeed effective.

TIME COMPLEXITY

Each operation is taken from a dictionary, so adding each feature can be done
in constant, O(1) time.

e) Arc-eager shift-reduce parser
The running time for arc-eager shift reduce parser is linear in the number of
words of the sentence, that is, O(n), because the number of steps is at most
2m + 2, and the Oracle takes only constant time O(1) to give the value of the
transistion.

Though it runs in linear time, in some cases, we may get unconnected components
in the dependency graph. Also, this is a greedy strategy and may not always
return the best solution.
