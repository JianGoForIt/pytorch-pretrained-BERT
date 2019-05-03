# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from smallfry import compress
from smallfry import utils


logger = logging.getLogger(__name__)
config = {}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def get_processor(task_name):
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
    }
    return processors[task_name]

def get_output_mode(task_name):
    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
    }
    return output_modes[task_name]

def get_upper_case_task_name(task_name):
    upper_case_tasks = {
        "cola": "CoLA",
        "mnli": "MNLI",
        "mrpc": "MRPC",
        "sst-2": "SST-2",
        "sts-b": "STS-B",
        "qqp": "QQP",
        "qnli": "QNLI",
        "rte": "RTI",
        "wnli": "WNLI",
    }
    return upper_case_tasks[task_name]

def compress_embeddings(X, b, compress_type, seed):
    logger.info('Beginning to compress embeddings')
    if compress_type == 'uniform':
        Xq, frob_squared_error, elapsed = compress.compress_uniform(X, b, adaptive_range=True)
    elif compress_type == 'kmeans':
        Xq, frob_squared_error, elapsed = compress.compress_kmeans(X, b, random_seed=seed)
    elif compress_type == 'pca':
        pca_dim = int(X.shape[1] * b / 32.0)
        Xq, frob_squared_error, elapsed = compress.compress_pca(X, pca_dim, keep_v=True)
    else:
        raise Exception('Other compress types not yet supported.')
    logger.info('Done compressing embeddings. Elapsed = {}, Frob-squared-error = {}'.format(elapsed, frob_squared_error))
    return Xq, frob_squared_error, elapsed

def init_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=['cola', 'mnli', 'mrpc', 'sst-2', 'sts-b', 'qqp', 'qnli', 'rte', 'wnli'],
                        help="The name of the task to train.")
    parser.add_argument("--rungroup",
                        default=None,
                        type=str,
                        required=True,
                        help="The run group for organizing results.")

    ## Important parameters
    parser.add_argument("--bert_model",
                        default="bert-base-cased",
                        type=str,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased",
                                 "bert-base-multilingual-uncased", "bert-base-multilingual-cased", "bert-base-chinese"],
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--data_dir",
                        default='/proj/smallfry/glue_data',
                        type=str,
                        help="The base directory of the input data. Data files should be under data_dir/task-name folder.")
    parser.add_argument("--output_dir",
                        default='/proj/smallfry/results/glue/',
                        type=str,
                        help="The base directory for where the model predictions and checkpoints will be written. "
                             "Results will be written to output_dir/task-name/run-group/run-name folder.")
    parser.add_argument("--git_repo_dir",
                        default='/proj/smallfry/git/smallfry',
                        type=str,
                        help="The directory of the git repo. Used for getting git hash and diff strings.")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument("--checkpoint_metric",
                        default='eval_loss_min',
                        type=str,
                        help="The metric to use to determine best performing epoch. Append '_min'/'_max' to metric name if minimium/maximum value determines best epoch.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    # Compression parameters
    parser.add_argument('--freeze_embeddings',
                        action='store_true',
                        help="Specifies if to freeze the WordPiece embeddings in the BERT model during training.")
    parser.add_argument('--compresstype',
                        type=str,
                        default='nocompress',
                        choices=['nocompress','uniform','kmeans','pca','dca','tt'],
                        help='Name of compression method to use.')
    parser.add_argument('--bitrate',
                        type=float,
                        default=32,
                        help='The number of bits per entry of embedding matrix.')

    # Other parameters
    parser.add_argument("--cache_dir",
                        default='/proj/smallfry/bert_pretrained_models',
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser

def get_filename(suffix):
    return os.path.join(config['output_dir'], config['full_run_name'] + suffix)

def validate_config():
    if config['compresstype'] != 'nocompress' and not config['freeze_embeddings']:
        raise ValueError('Can only do compression if freezing embeddings.')

    if config['compresstype'] != 'nocompress' and config['bitrate'] >= 32:
        raise ValueError('If compressing, must specify bitrate < 32.')

    if config['gradient_accumulation_steps'] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            config['gradient_accumulation_steps']))

def init_config(parser):
    global config
    config = vars(parser.parse_args())
    orig_config = config.copy()
    validate_config()

    # 1) Set up run directory and run names
    config['data_dir'] = os.path.join(config['data_dir'], get_upper_case_task_name(config['task_name']))
    config['rungroup'] = '{}-{}'.format(utils.get_date_str(), config['rungroup'])
    config['short_run_name'] = 'freeze,{}_compresstype,{}_bitrate,{}_seed,{}'.format(
        config['freeze_embeddings'], config['compresstype'], config['bitrate'], config['seed'])
    config['full_run_name'] = '{}_task,{}_{}'.format(config['rungroup'], config['task_name'], config['short_run_name'])
    config['output_dir'] = os.path.join(config['output_dir'], config['task_name'], config['rungroup'], config['short_run_name'])
    utils.ensure_dir(config['output_dir']) # Make the output directory if it doesn't exist

    # 2) Add important entries into final config dictionary
    git_hash, git_diff = utils.get_git_hash_and_diff(config['git_repo_dir'], log=False)
    config['git_hash'] = git_hash
    config['git_diff'] = git_diff
    config['train_batch_size'] = config['train_batch_size'] // config['gradient_accumulation_steps']

    # 3) save the original config, and the current config.
    utils.save_to_json(orig_config, get_filename('_orig_config.json'))
    utils.save_to_json(config, get_filename('_config.json'))

def init_logging(log_filename):
    # Log to file in output directory as well as to stdout.
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.DEBUG,
                        handlers=[
                            logging.FileHandler(log_filename, mode='w'),
                            logging.StreamHandler()])

def init_random_seeds(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def init():
    parser = init_parser()
    init_config(parser)
    init_logging(get_filename('.log'))
    device = torch.device('cuda' if torch.cuda.is_available() and not config['no_cuda'] else 'cpu')
    n_gpu = torch.cuda.device_count()
    init_random_seeds(config['seed'], n_gpu)
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    return device, n_gpu

def get_data():
    processor = get_processor(config['task_name'])()
    train_examples = processor.get_train_examples(config['data_dir'])
    eval_examples = processor.get_dev_examples(config['data_dir'])
    label_list = processor.get_labels()
    return train_examples, eval_examples, label_list

def get_num_train_optimization_steps(num_train_examples):
    num_train_optimization_steps = int(
        num_train_examples / config['train_batch_size'] / config['gradient_accumulation_steps']) * config['num_train_epochs']
    return num_train_optimization_steps

def get_optimizer(model, num_train_examples):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config['learning_rate'],
                         warmup=config['warmup_proportion'],
                         t_total=get_num_train_optimization_steps(num_train_examples))
    return optimizer

def freeze_and_compress_embeddings(model, device):
    X = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy().copy()
    dummy_word_list = ['x'] * X.shape[0]
    utils.save_embeddings(get_filename('_orig_embeddings.txt'), X, dummy_word_list)
    Xq = X
    compression_results = None
    # Freeze and perhaps compress embeddings
    if config['freeze_embeddings']:
        # "freeze" WordPiece embbedings by setting requires_grad to False.
        model.bert.embeddings.word_embeddings.weight.requires_grad = False
        # perform compression of the WordPiece embeddings.
        if config['compresstype'] != 'nocompress':
            Xq,_,elapsed = compress_embeddings(X, config['bitrate'], config['compresstype'], config['seed'])
            # copy compressed WordPiece embeddings into the BERT model.
            model.bert.embeddings.word_embeddings.weight.copy_(torch.from_numpy(Xq.copy()).to(device))
            # Measure compression quality (reconstruction error, PIP, deltas, overlap).
            compression_results = utils.compute_basic_compression_results(X, Xq)
            compression_results['elapsed'] = elapsed
    utils.save_embeddings(get_filename('_compressed_embeddings.txt'), Xq, dummy_word_list)
    return Xq, compression_results

def save_model_and_tokenizer(model, tokenizer):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(config['output_dir'], WEIGHTS_NAME)
    output_config_file = os.path.join(config['output_dir'], CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(config['output_dir'])

    # *** Avner removed the lines below. Unclear why they had this code in the original run_classifier.py. ***
    # # Load a trained model and vocabulary that you have fine-tuned
    # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
    # tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

def get_model(num_labels, device, n_gpu):
    model = BertForSequenceClassification.from_pretrained(config['bert_model'],
              cache_dir=config['cache_dir'],
              num_labels=num_labels)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model

def get_dataloader(examples, label_list, tokenizer, output_mode, train=True):
    batch_size = config['train_batch_size'] if train else config['eval_batch_size']
    features = convert_examples_to_features(
        examples, label_list, config['max_seq_length'], tokenizer, output_mode)
    logger.info("***** Running {} *****".format('training' if train else 'evaluation'))
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dtype = torch.long if output_mode == "classification" else torch.float
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=dtype)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = RandomSampler(data) if train else SequentialSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size), all_label_ids

def run_train_epoch(model, train_dataloader, optimizer, output_mode, n_gpu, device, num_labels):
    tr_loss = 0
    num_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        logits = model(input_ids, segment_ids, input_mask, labels=None)

        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if config['gradient_accumulation_steps'] > 1:
            loss = loss / config['gradient_accumulation_steps']

        loss.backward()
        tr_loss += loss.item()
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            num_steps += 1
    return tr_loss/num_steps

def run_evaluation(model, eval_dataloader, eval_label_ids, output_mode, device, num_labels):
    eval_loss = 0
    num_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        
        eval_loss += tmp_eval_loss.mean().item()
        num_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / num_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(config['task_name'], preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    return result

def update_full_results(full_results, result, epoch):
    is_best_epoch = False
    for k,v in result.items():
        if k not in full_results:
            full_results[k] = []
        full_results[k].append(v)
        if v == min(full_results[k]):
            full_results[k + '_min'] = v
            if k + '_min' == config['checkpoint_metric']:
                is_best_epoch = True
        if v == max(full_results[k]):
            full_results[k + '_max'] = v
            if k + '_max' == config['checkpoint_metric']:
                is_best_epoch = True
    if is_best_epoch:
        full_results['checkpoint'] = result
        full_results['best_epoch'] = epoch

def main():
    device, n_gpu = init()
    train_examples, eval_examples, label_list = get_data()
    output_mode = get_output_mode(config['task_name'])
    tokenizer = BertTokenizer.from_pretrained(
        config['bert_model'],
        do_lower_case=('uncased' in config['bert_model'])
    )

    # Prepare model and optimizer
    model = get_model(len(label_list), device, n_gpu)
    optimizer = get_optimizer(model, len(train_examples))

    # if config['freeze_embeddings'] is true, freeze and then optionally compress embeddings.
    Xq, compression_results = freeze_and_compress_embeddings(model, device)

    train_dataloader,_ = get_dataloader(train_examples, label_list, tokenizer, output_mode, train=True)
    full_results = {}
    if config['compresstype'] != 'nocompress':
        full_results['compression-results'] = compression_results
    for epoch in trange(int(config['num_train_epochs']), desc="Epoch"):
        # Do one epoch of training
        model.train()
        tr_loss = run_train_epoch(model, train_dataloader, optimizer, output_mode, n_gpu, device, len(label_list))

        # Run evaluation
        model.eval()
        eval_dataloader, eval_label_ids = get_dataloader(eval_examples, label_list, tokenizer, output_mode, train=False)
        result = run_evaluation(model, eval_dataloader, eval_label_ids, output_mode, device, len(label_list))
        result['train_loss'] = tr_loss
        update_full_results(full_results, result, epoch)
        utils.save_to_json(full_results, get_filename('_results.json'))

    # Assert that after training has completed, embeddings didn't change if config['freeze_embeddings'] is True
    if config['freeze_embeddings']:
        assert np.array_equal(Xq, model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()), 'Embeddings changed during training.'

    # Save model, tokenizer, final results, and final config.
    save_model_and_tokenizer(model, tokenizer)
    utils.save_to_json(full_results, get_filename('_final_results.json'))
    utils.save_to_json(config, get_filename('_final_config.json'))
    logging.info('Run complete. Exiting compress.py main method.')

if __name__ == "__main__":
    main()
