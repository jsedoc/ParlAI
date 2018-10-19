#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Module files as torch.nn.Module subclasses for Seq2seqAgent."""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from parlai.core.utils import NEAR_INF


class EncoderRNN(nn.Module):
    """Encodes the input context."""

    def __init__(self, input_size, hidden_size, numlayers):
        """Initialize encoder.

        :param input_size: size of embedding/how big vocab is
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=numlayers,
                          batch_first=True)

    def forward(self, input, hidden=None):
        """Return encoded state.

        :param input: (batchsize x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, output_size, hidden_size, numlayers):
        """Initialize decoder.

        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=numlayers,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        """Return encoded state.

        :param input: batch_size x 1 tensor of token indices.
        :param hidden: past (e.g. encoder) hidden state
        """
        emb = self.embedding(input)
        rel = F.relu(emb)
        output, hidden = self.gru(rel, hidden)
        scores = self.softmax(self.out(output))
        return scores, hidden
