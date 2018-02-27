# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import maintain_dialog_history, PaddingUtils


import torch
from torch.autograd import Variable
from torch import optim, cuda
import torch.nn as nn

from collections import deque

import os
import math
import random
import glob
import sys

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


# CODE: make them private members of the class
def check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)

class OpennmtAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    OpenNMT.

    For more information, see `OpenNMT <http://opennmt.net>`_.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        # opts.py
        opts.add_md_help_argument(argparser)
        opts.model_opts(argparser)
        opts.train_opts(argparser)
        opt = argparser.parse_args()

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = objectview(self.opt)  # there is a deepcopy in the init

        if shared:
            print('Not sure if shared works.')
            #raise AssertionError('OpenNMT agent cannot run in shared mode.')
        if opt.word_vec_size != -1:
            opt.src_word_vec_size = opt.word_vec_size
            opt.tgt_word_vec_size = opt.word_vec_size

        if opt.layers != -1:
            opt.enc_layers = opt.layers
            opt.dec_layers = opt.layers

        opt.brnn = (opt.encoder_type == "brnn")
        if opt.seed > 0:
            random.seed(opt.seed)
            torch.manual_seed(opt.seed)

        if opt.rnn_type == "SRU" and not opt.gpuid:
            raise AssertionError("Using SRU requires -gpuid set.")

        if torch.cuda.is_available() and not opt.gpuid:
            print("WARNING: You have a CUDA device, should run with -gpuid 0")

        if opt.gpuid:
            cuda.set_device(opt.gpuid[0])
            if opt.seed > 0:
                torch.cuda.manual_seed(opt.seed)

            if len(opt.gpuid) > 1:
                sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
                sys.exit(1)

        if opt.tensorboard:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(opt.tensorboard_log_dir, comment="Onmt")

        # Load checkpoint if we resume from a previous training.
        if opt.train_from:
            print('Loading checkpoint from %s' % opt.train_from)
            checkpoint = torch.load(opt.train_from,
                                    map_location=lambda storage, loc: storage)
            model_opt = checkpoint['opt']
            # I don't like reassigning attributes of opt: it's not clear.
            opt.start_epoch = checkpoint['epoch'] + 1
        else:
            checkpoint = None
            model_opt = opt
            
        # NOTE: TBD
        # # Peek the fisrt dataset to determine the data_type.
        # # (All datasets have the same data_type).
        # first_dataset = next(lazily_load_dataset("train"))
        # data_type = first_dataset.data_type
        data_type = 'text'

        # # Load fields generated from preprocess phase.
        # fields = load_fields(first_dataset, data_type, checkpoint)
        
        # # Report src/tgt features.
        # collect_report_features(fields)

        # # Build model.
        # model = build_model(model_opt, opt, fields, checkpoint)
        # tally_parameters(model)
        # check_save_model_path()

        # # Build optimizer.
        # optim = build_optim(model, checkpoint)
        
        # train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
        # valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
        
        # trunc_size = opt.truncated_decoder  # Badly named...
        # shard_size = opt.max_generator_batches
        # norm_method = opt.normalization
        # grad_accum_count = opt.accum_count

        # self.trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
        #                        trunc_size, shard_size, data_type,
        #                        norm_method, grad_accum_count)

        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        # NOTE: JS - TBD need to add more ... this is from seq2seq.py
        # model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
        #               'encoder', 'decoder', 'lookuptable', 'attention',
        #               'attention_length'}

        model_args = {
            'src_word_vec_size',
            'tgt_word_vec_size',
            'word_vec_size',
            'share_decoder_embeddings',
            'share_embeddings'
        }
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def observe(self, observation):
        """Save observation for act.
        """
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        print(observation)
        # NOTE: batch_idx not currently used.
        batch_idx = self.opt.get('batchindex', 0)
        if not self.episode_done and not observation.get('preprocessed', False):
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def predict(self, xs, ys=None, cands=None, valid_cands=None):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        is_training = ys is not None
        text_cand_inds, loss_dict = None, None
        if is_training:
            print('TRAINING!!!')
        else:
            pass
        predictions = xs
        text_cand_inds = cands
        loss_dict = {}
        return predictions, text_cand_inds, loss_dict

    def batch_act(self, observations):
        batchsize = len(observations)
        print("Batch size =", batchsize)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]
        self.predict([1,1,1],[2,3,3])

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        """Save model parameters if model_file is set."""


    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            states = torch.load(read)

        return states['opt'], states
