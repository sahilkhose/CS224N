#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size): 
    	""" Init Highway network.

    	@param embed_size (int): Word Embedding size (dimensionality)
    	@param dropout_rate (float): Dropout probability, for highway
    	"""

    	super(Highway, self).__init__()
    	self.word_embed_size = word_embed_size

    	# default values
    	self.proj = None
    	self.relu = None
    	self.gate = None
    	self.sigmoid = None
    	self.word_emb = None

    	"""TODO - Initialize the following variables:
    		self.proj (Linear Layer with bias)
    		self.relu (ReLU function)
    		self.gate (Linear Layer with bias)
    		self.sigmoid (Sigmoid function)
    		self.word_emb (Dropout Layer)
    	"""

    	self.proj = nn.Linear(in_features=self.word_embed_size, out_features=self.word_embed_size, bias=True)
    	self.relu = nn.ReLU()

    	self.gate = nn.Linear(in_features=self.word_embed_size, out_features=self.word_embed_size, bias=True)
    	self.sigmoid = nn.Sigmoid()


    def forward(self, X_conv_out: torch.Tensor):
    	"""Take a mini-batch input from the 1-D convolution, output the word embedding 
    	   after passing the input through a highway network. 

    	@param X_conv_out (Tensor): tensor of shape (b, word_emb) 
    							coming from the 1-D conv output
    							b = batch size
		
		@returns X_word_emb (Tensor) : tensor of shape (b, word_emb) which is the 
										final word embedding
    	"""

    	X_proj = self.relu(self.proj(X_conv_out))
    	X_gate = self.sigmoid(self.gate(X_conv_out))
    	X_highway = X_gate * X_proj + (1 - X_gate) * X_conv_out

    	return X_highway



    ### END YOUR CODE

