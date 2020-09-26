#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embed_size, word_embed_size, kernel_size=5, padding=1):
        """ Init 1-D Conv Network.

        @param kernel_size (int): Kernel size for 1-D convolutions
        @param padding (int): Padding size
        """

        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size

        # default values
        self.conv = None 
        self.maxpool = None
        self.relu = None

        """
        TODO - Initialize the following variables:
            self.conv (Conv1d Layer)
            self.maxpool (MaxPool1d Layer)
            self.relu (ReLU function)
        """

        self.conv = nn.Conv1d(in_channels=self.char_embed_size, out_channels=self.word_embed_size, 
                                kernel_size=self.kernel_size, padding=self.padding)
        # self.maxpool = nn.MaxPool1d(kernel_size=)
        self.relu = nn.ReLU()

    def forward(self, X_reshaped: torch.Tensor):
        """
        Take a mini-batch input from the padded character indices and
        return the output after passing the input through a Conv1d network

        @param X_reshaped (Tensor): tensor of shape (b, e_char, m_word)
                                b = batch size

        @returns X_conv_out (Tensor): tensor of shape (b, e_char)
                                    this is sent to the highway network
        """

        X_conv = self.conv(X_reshaped)
        # X_conv_out = self.maxpool(self.relu(X_conv))
        X_conv_out = torch.max(self.relu(X_conv), dim=2)[0]

        return X_conv_out

    ### END YOUR CODE

