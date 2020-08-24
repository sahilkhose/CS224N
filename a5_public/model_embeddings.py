#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.vocab = vocab
        self.size_of_vocab = len(self.vocab.char2id)
        self.e_char = 50
        self.word_embed_size = word_embed_size


        self.embeddings = nn.Embedding(num_embeddings=self.size_of_vocab, embedding_dim=self.e_char)
        self.cnn = CNN(char_embed_size=self.e_char, word_embed_size=self.word_embed_size, kernel_size=5)
        self.highway = Highway(word_embed_size=self.word_embed_size)
        self.dropout = nn.Dropout(0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        sentence_length, batch_size, max_word_length = input.shape
        X_emb = self.embeddings(input) # (sentenece_length, b, m_word, e_char)

        X_reshaped = X_emb.permute(0, 1, 3, 2) # (sentence_length, b, e_char, m_word)

        X_conv = self.cnn(X_reshaped.view(-1, self.e_char, max_word_length)) # (sentence_len * b, e_word)
 
        X_highway = self.highway(X_conv) # (sentence_len * b, e_word)

        X_word_emb = self.dropout(X_highway.view(sentence_length, batch_size, self.word_embed_size))

        return X_word_emb

        ### END YOUR CODE

