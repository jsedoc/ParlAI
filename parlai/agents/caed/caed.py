#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Example sequence to sequence agent for ParlAI "Creating an Agent" tutorial.
http://parl.ai/static/docs/tutorial_seq2seq.html
"""

#from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from .modules import Seq2seq, opt_to_kwargs

import torch
import torch.nn as nn

import spacy, tqdm
import collections
from collections import defaultdict
import concurrent.futures
import requests, json
import zlib
import datetime
# from ccg_nlpy import remote_pipeline


class CaedAgent(TorchGeneratorAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model is based on Sean Robertson's `seq2seq tutorial
    <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('CAED Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=1,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot',
                                    'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seq.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
                           help='Probability of replacing tokens with UNK in training.')
        super(cls, CaedAgent).add_cmdline_args(argparser)
        CaedAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """Return current version of this model, counting up from 0.

        """
        return 0

    def __init__(self, opt, shared=None):
        """Initialize example seq2seq agent.

        :param opt: options dict generated by parlai.core.params:ParlaiParser
        :param shared: optional shared dict with preinitialized model params
        """
        super().__init__(opt, shared)
        self.id = 'Caed'
        # store latest tmdb data
        self.tmdb_ent_dict = self.get_tmdb_ids()

    def build_model(self, states=None):
        """Initialize model, override to change model setup."""
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        self.model = Seq2seq(
            len(self.dict), opt['embeddingsize'], opt['hiddensize'],
            padding_idx=self.NULL_IDX, start_idx=self.START_IDX,
            end_idx=self.END_IDX, unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs)

        if (opt.get('dict_tokenizer') == 'bpe' and
                opt['embedding_type'] != 'random'):
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.decoder.lt.weight,
                                  opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(self.model.encoder.lt.weight,
                                      opt['embedding_type'], log=False)

        if states:
            # set loaded states if applicable
            self.model.load_state_dict(states['model'])

        if self.use_cuda:
            self.model.cuda()

        if opt['embedding_type'].endswith('fixed'):
            print('Seq2seq: fixing embedding weights.')
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                self.model.decoder.e2s.weight.requires_grad = False

        self.model.nlp = spacy.load('en_core_web_lg')


        if self.use_cuda:
            self.model.cuda()
            if self.multigpu:
                self.model = torch.nn.DataParallel(self.model)
                self.model.encoder = self.model.module.encoder
                self.model.decoder = self.model.module.decoder
                self.model.longest_label = self.model.module.longest_label
                self.model.output = self.model.module.output
                self.model.reorder_encoder_states = (
                    self.model.module.reorder_encoder_states
                )

        return self.model

    def get_tmdb_ids(self):
        ent_dict = dict()
        date = (datetime.datetime.now() - datetime.timedelta(1)).strftime('%m_%d_%Y')
        person_input_file = 'http://files.tmdb.org/p/exports/person_ids_' + date + '.json.gz'
        movie_input_file = 'http://files.tmdb.org/p/exports/movie_ids_' + date + '.json.gz'
        r_person = requests.get(person_input_file)
        r_movie = requests.get(movie_input_file)

        decompressed_data = zlib.decompress(r_person.content, 16+zlib.MAX_WBITS)
        decompressed_data = decompressed_data.decode("utf-8")
        for ent in decompressed_data.split('\n'):
            if ent != None and ent != '\n' and ent.strip() != '':
                ent_dict[json.loads(ent)['name']] = 'PER'

        decompressed_data = zlib.decompress(r_movie.content, 16+zlib.MAX_WBITS)
        decompressed_data = decompressed_data.decode("utf-8")
        for ent in decompressed_data.split('\n'):
            if ent != None and ent != '\n' and ent.strip() != '':
                ent_dict[json.loads(ent)['original_title']] = 'WORK_OF_ART'

        return ent_dict

    def process_line(self, text):

        # first preprocess using database of movie labels

        target_ents = ["ORG", "PERSON"]

        l = text

        if "__null__" in text:
        	return "__null__"

        print('\n')
        print('ORIGINAL TEXT :' + text + '\n')

        word_to_ent = dict()
        ent_counts = dict()

        spdoc = self.model.nlp(l)

        for word in self.tmdb_ent_dict:
            if word in text:
                word_to_ent[word] = self.tmdb_ent_dict[word]

        for e in spdoc.ents:
            #  word -> entity
            word = e.as_doc().text.strip()
            if word not in word_to_ent:
              word_to_ent[word] = e.label_

        for word in word_to_ent:
            ent = word_to_ent[word]
            previous_count = 0
            if ent in ent_counts:
                previous_count = ent_counts[ent]
            ent_counts[ent] = previous_count + 1
            word_to_ent[word] = word_to_ent[word] + '@' + str(ent_counts[ent])

        for word in word_to_ent:
          if (word != '' and word not in self.tmdb_ent_dict):
            text = text.replace(word, word_to_ent[word])

        print('CONVERTED TEXT :' + text + '\n')
        return text


    def v2t(self, vector):
        """Convert vector to text.

        :param vector: tensor of token indices.
            1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == self.END_IDX:
                    break
                else:
                    output_tokens.append(token)
            return self.process_line(self.dict.vec2txt(output_tokens))
        elif vector.dim() == 2:
            return [self.v2t(vector[i]) for i in range(vector.size(0))]
        raise RuntimeError('Improper input to v2t with dimensions {}'.format(
            vector.size()))

    # we can change vectorize to use spacey
    def vectorize(self, *args, **kwargs):
        """Call vectorize without adding start tokens to labels."""
        kwargs['add_start'] = False

        # (Pdb) p args[0]['text']
        # 'your persona: i like to remodel homes.\nyour persona: i like to go hunting.\nyour persona: i like to shoot a bow.\nyour persona: my favorite holiday is halloween.\nhi , how are you doing ? i am getting ready to do some cheetah chasing to stay in shape .'
        # (Pdb) p args[0]['labels']
        # ('you must be very fast . hunting is one of my favorite hobbies .',)
        args[0]['text'] = self.process_line(args[0]['text'])

        return super().vectorize(*args, **kwargs)

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            if self.multigpu:
                model['model'] = self.model.module.state_dict()
            else:
                model['model'] = self.model.state_dict()
            model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file
            with open(path + '.opt', 'w') as handle:
                # save version string
                self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        if self.multigpu:
            self.model.module.load_state_dict(states['model'])
        else:
            self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states
