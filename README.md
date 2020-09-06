----------------------------------------------------------------------------------------
<!-- <p align="center"><img width="40%" src="https://github.com/sahilkhose/CS224N/blob/master/stanford-cs224n-course-header.jpg" /></p> -->

-------------------------------------------------------------------------------------------

# CS224n: Natural Language Processing with Deep Learning (Stanford / Winter 2020)
*This Repository Contains Solution to the Practical Assignments of the [CS224n (Natural Language Processing with Deep Learning)](http://web.stanford.edu/class/cs224n/) offered by Stanford University at Winter 2020 Taught by [Christopher Manning](https://nlp.stanford.edu/~manning/)*

--------------------------------------------------------------------------------------------

## Assignments
1. [Assignment 1: Exploring Word Vectors](https://github.com/sahilkhose/CS224N/tree/master/a1)
2. [Assignment 2: Word2Vec from Scratch](https://github.com/sahilkhose/CS224N/tree/master/a2)
3. [Assignment 3: Neural Dependency Parser](https://github.com/sahilkhose/CS224N/tree/master/a3/student)
4. [Assignment 4: Seq2Seq Machine Translation](https://github.com/sahilkhose/CS224N/tree/master/a4)<br>
    **Corpus BLEU: 35.8780** es-en translation
5. [Assignment 5: Hybrid Word-Character Seq2Seq Machine Translation](https://github.com/sahilkhose/CS224N/tree/master/a5_public)<br>
   **Corpus BLEU: 37.0347** es-en translation

-------------------------------------------------------------------------------------------------------------
<br>

## Assignment 1. Word Embeddings
This assignments has two parts which are about representing words with dense vectors. Having these vectors can be really useful in down-stream tasks in NLP. The first method of deriving weord vector stems from the co-occurence matrices and SVD decomposition. The second method is based on maximum-likelihood training in ML.

### 1. Count-Based Word Vectors
In this part, you have to use the co-occurence matrices to develop dense vectors for words. A co-occurence matrix counts how often different terms co-occur in different documents. To derive a co-occurence matrix, we use a window with a fixed size _w_, and then slide this window over all of the documents. Then, we count how many times two different words v_i and v_j occurs with each other in a window, and put this number in the (i, j) entry of the matrix.<br/>
Then, we have to run dimensionality reduction on the co-occurence matrix using singular value decomposition. We then select the top _r_ components after the decomposition and thus, derive r-dimensional embeddings for words.


<p align="center">
<img src="/https://github.com/sahilkhose/CS224N/blob/master/figures/svd.jpg" alt="drawing" width="400"/>
</p>



### 2. Prediction(or Maximum Likelihood)-Based Word Vectors: Word2Vec
In this part, you will work with the pretrained word2vec embeddings of [gensim](https://radimrehurek.com/gensim/) package. There are lots of tasks in this part. At first, you have to reduce the dimensionality of word vectors using SVD from 300 to 2 so as to be able to visualize the vectors and analyze this visualization. Then you will find the closest word vectors to a given word vector. You will get to know words with several meanings (Polysemous words). You will get to know the analogy task, mentioed for the first time in the original paper of word2vec [(Mikolov et al. 2013)](https://arxiv.org/pdf/1301.3781.pdf%5D). The task is simple: given words x, y, and z, you have to find a word w such that the following relationship holds: x to y is like z to w. For example, Rome to Italy is like D.C. to the United Stats. You will find that solving this task with word2vec vectors is easy and is just a simple addition and subtraction of vectors, which is a nice feature of word2vec.


<p align="center">
<img src="/https://github.com/sahilkhose/CS224N/blob/master/figures/analogy.jpg" alt="drawing" width="400"/>
</p>



## Assignment 2. Word2Vec from Scratch
In this assignment you will get familiar with the word2vec algorithm. The key insight behind word2vec is that "a word is known by the company it keeps". There are two models introduced by the word2vec paper working based on this idea: Skip-gram and Continuous Bag Of Words (CBOW). In this assignment you have to implement Skip-gram model with Numpy from scratch. You have to implement the both version of Skipgram; the first one is with the naive softmax loss and the second one, which is much faster, is with the negative sampling loss. You have to implement both the forward and backward passes of the two versions of model from scratch. Your implementation of the first version is just sanity-checked on a small dataset, but you have to run the second version on the Stanford Sentiment Treebank which takes roughly an hour. I highly recommend everyone who is willing to gain a deep understanding of word2vec to first do the theoretical part of this assignment (available [here](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a2.pdf)) and do the practical part afterwards.


<p align="center">
<img src="/https://github.com/sahilkhose/CS224N/blob/master/figures/word2vec.jpg" alt="drawing" width="450"/>
</p>



## Assignment 3. Neural Dependency Parsing
If you have take a compiler course before, you have definitely heard the term "parsing". This assignment is about "dependency parsing" where you have to train a model that can sepcify the dependencir . If you remember "Shift-Reduce Parser" from your Compiler class, then you will find the ideas here quite familiar. The only difference is that we use a neural network to find the dependencies. 

In the theretical part of assignment (handout is available [here](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a3.pdf)), the Adam optimizer is first introduced and you have to answer some questions about this optimizer. Then, there is a question about Dropout as a regularization technique. Both of Adam optimizer and Dropout will be used in the neural dependency parser you are going to implement with PyTorch.<br/>
The parser will do one of the following three moves: 1) Shift 2) Left-arc 3) Right-arc. You can read more about the details of these three moves in the handout of the assignment. What you network should do is to predict one of these moves at every step. For predicting each move, your model needs features which are going to be extracted from the stack and buffer of each stage (there is a stack and a buffer throught parsing which let you know what you have already parsed and what is still remaining for parsing). The good news is that the code for extracting features is given to you so as to help you just focus on the neural network part! There are lots of hints throughout the assignment --as this is the first assignment in the course where students work with PyTorch-- that walk you through implementing each part. 

<p align="center">
<img src="/https://github.com/sahilkhose/CS224N/blob/master/figures/dependency-parsing.jpg" alt="drawing" width="450"/>
</p>





## Assignment 4. Seq2Seq Machine Translation
In my opinion, this assignment is the most importatnt assignment of the course. Generally, you have to implement a Seq2Seq model that translates German sentences into English. The model that you will implement is based on [Luong et al. 2015](https://arxiv.org/pdf/1508.04025.pdf) . You will some important practical notes, such as working with recurrent neural networks in PyTorch, learning the differences between training and test time in RNNs, and implementing attention mechanism, and etc. The pipeline and the implementations provided for you are standard and inspired by the [Open-NMT](https://github.com/OpenNMT/OpenNMT-py) package. I highly recommend you to not just implement what is left for you and go further and evaluate carefully what TA's have provided for you, from getting inputs from CLI to evaluation metrics of NMT models and algorithms used for the decoding stage of RNNs such as Beam Search. There are lots of PyTorch techniques and functions that you can grasp and use in your future projects.

<p align="center">
<img src="/https://github.com/sahilkhose/CS224N/blob/master/figures/nmt.jpg" alt="drawing" width="350"/>
</p>





## Assignment 5. Hybrid Word-Character Seq2Seq Machine Translation
The idea behind this assignment is same as the previous assingments, except that the model becomes more powerful as we will combine character-level with word-level language modeling. The idea is that whenever the NMT model from assignment 4 generates an <unk> token we do not put it in the output. Instead, we run a character-level language model and generate a word in the output character by character. In fact, this hybrid word-cahracter approach was proposed by [Luong and Manning 2016](https://arxiv.org/pdf/1604.00788.pdf) and tunred out to be effective in increasing the performance of the NMT model.


<p align="center">
<img src="/https://github.com/sahilkhose/CS224N/blob/master/figures/nmt-hybrid.jpg" alt="drawing" width="350"/>
</p>
