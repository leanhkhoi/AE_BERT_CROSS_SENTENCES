# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
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
import copy
import os
import logging
import argparse
import random
import json
import time
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import BertPreTrainedModel, BertModel, \
    BertLayer as OfficialBertLayer, AdamW, get_linear_schedule_with_warmup
from common import get_labels, get_predictions, \
    get_predictions2, write_result, PreprocessConfig, combine_sentences2, predict, \
    read_preprocess_load, LoaderConfig, convert_to_features, to_data_loader
from absa_data_utils import ABSATokenizer
import modelconfig
from torchcrf import CRF

logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GRoIE(nn.Module):
    def __init__(self, count, config, num_labels):
        super(GRoIE, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.crf_layers = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        for i in range(count):
            self.pre_layers.append(OfficialBertLayer(config))
            self.crf_layers.append(CRF(num_labels))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i - 1], attention_mask)[0]
            layer = self.dropout(layer)
            logits = self.classifier(layer)
            if labels is not None:
                loss = self.crf_layers[i](logits.view(128, -1, self.num_labels), labels.view(128, -1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0) / self.count
        return -total_loss, avg_logits


class BertForAE(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BertForAE, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.groie = GRoIE(4, config, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        out_puts = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True,
                             output_attentions=True)
        loss, logits = self.groie(out_puts.hidden_states,
                                  self.get_extended_attention_mask(attention_mask, input_ids.size(), None), labels)
        if labels is not None:
            return loss
        else:
            return logits


def train(args):
    label_list = get_labels()
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], do_basic_tokenize=False)

    preprocess_config = PreprocessConfig(tokenizer, seq_len=args.max_seq_length, no_context=args.no_context,
                                         seq_start=args.seq_start)
    vectorize_config = LoaderConfig(batch_size=args.train_batch_size, sampler=RandomSampler)
    train_dataloader, train_data = read_preprocess_load(os.path.join(args.data_dir, "train.json"),
                                                        preprocess_config, vectorize_config)

    num_train_steps = int(len(train_data.sentences) / args.train_batch_size) * args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data.sentences))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    # >>>>> validation
    if args.no_valid is False:
        vectorize_config = LoaderConfig(batch_size=args.train_batch_size, sampler=SequentialSampler)
        valid_dataloader, valid_data = read_preprocess_load(os.path.join(args.data_dir, "dev.json"),
                                                            preprocess_config, vectorize_config)
        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_data.sentences))
        logger.info("  Batch size = %d", args.train_batch_size)

        best_valid_loss = float('inf')
        valid_losses = []
    # <<<<< end of validation declaration

    model = BertForAE.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], num_labels=len(label_list))
    model.to(device)
    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # t_total = num_train_steps
    ### In 🤗 Transformers, optimizer and schedules are split and instantiated like this:
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.01,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_train_steps / 10),
                                                num_training_steps=num_train_steps)  # PyTorch scheduler
    global_step = 0
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch

            loss = model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           1.0)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            global_step += 1
            # >>>> perform validation at the end of each epoch.
        print("training loss: ", loss.item(), epoch + 1)
        new_dirs = os.path.join(args.output_dir, str(epoch + 1))
        os.mkdir(new_dirs)
        if args.no_valid is False:
            model.eval()
            with torch.no_grad():
                losses = []
                valid_size = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                    input_ids, segment_ids, input_mask, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    losses.append(loss.data.item() * input_ids.size(0))
                    valid_size += input_ids.size(0)
                valid_loss = sum(losses) / valid_size
                logger.info("validation loss: %f, epoch: %d", valid_loss, epoch + 1)
                valid_losses.append(valid_loss)
                test(args, dev_as_test=True, output_dir=new_dirs, model=model)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            model.train()

    if args.no_valid is False:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)

    torch.save(model, os.path.join(args.output_dir, "model.pt"))


def test(args, dev_as_test=None, output_dir=None, model=None,
         model_dir=None):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)
    if output_dir is None:
        output_dir = args.output_dir
    if model_dir is None:
        model_dir = args.output_dir

    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    if dev_as_test:
        data_dir = os.path.join(args.data_dir, 'dev_as_test')
    else:
        data_dir = args.data_dir

    preprocess_config = PreprocessConfig(tokenizer, seq_len=args.max_seq_length, no_context=args.no_context,
                                         seq_start=args.seq_start)
    vectorize_config = LoaderConfig(batch_size=args.eval_batch_size)
    eval_dataloader, eval_data = read_preprocess_load(os.path.join(data_dir, "test.json"),
                                                      preprocess_config, vectorize_config)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data.sentences))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if model is None:
        _model = torch.load(os.path.join(model_dir, "model.pt"))
        _model.to(device)
        _model.eval()
    else:
        _model = model

    probs = np.array(predict(_model, eval_dataloader, device))
    preds = np.argmax(probs, axis=-1)

    pr_ensemble, pr_test_first = get_predictions(preds, eval_data.sentences, eval_data.sentence_numbers)
    prob_ensemble, prob_test_first = get_predictions2(probs, eval_data.sentences, eval_data.sentence_numbers)

    ens = [pr_ensemble, prob_ensemble, pr_test_first, prob_test_first]
    method_names = ['CMV', 'CMVP', 'F', 'FP']
    for ensem, method_name in zip(ens, method_names):
        output_eval_json = os.path.join(output_dir, "predictions_{}.json".format(method_name))
        write_result(output_eval_json, eval_data.orig_sentences, eval_data.lengths,
                     eval_data.sentences, eval_data.labels, ensem)

    if args.sentence_in_context:
        seq_len = args.max_seq_length
        tag_map = {l: i for i, l in enumerate(get_labels())}
        starting_pos = np.arange(0, seq_len, 32)
        #starting_pos[0] = 1
        for start_p in starting_pos:
            tt_lines, tt_tags, line_nos, line_starts = combine_sentences2(eval_data.sentences, eval_data.labels,
                                                                          seq_len - 1, start_p)

            input_ids, segment_ids, masks, label_ids = convert_to_features(tt_lines, tt_tags, tag_map, tokenizer,
                                                                           seq_len)
            data_loader = to_data_loader(input_ids, segment_ids, masks, label_ids, batch_size=args.eval_batch_size)
            probs = np.array(predict(_model, data_loader, device))
            preds = np.argmax(probs, axis=-1).tolist()

            pred_tags = []
            for i, pred in enumerate(preds):
                idx = line_nos[i].index(i)
                pred_tags.append([t for t in
                                  pred[line_starts[i][idx] + 1 :line_starts[i][idx] + 1 + len(eval_data.sentences[i])]])

            output_eval_json = os.path.join(output_dir, "predictions_start_position_{}.json".format(start_p))
            write_result(output_eval_json, eval_data.orig_sentences, eval_data.lengths,
                         eval_data.sentences, eval_data.labels, pred_tags)

    print("##############")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert-base', type=str)

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir containing json files.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--no_valid",
                        default=False,
                        action='store_true',
                        help="Whether to run validation.")
    parser.add_argument("--only_test",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_save",
                        default=False,
                        action='store_true',
                        help="Whether to save model after training.")
    parser.add_argument("--no_context",
                        default=False,
                        action='store_true',
                        help="Whether we're training a sentence-crossed model.")
    parser.add_argument("--sentence_in_context",
                        default=False,
                        action='store_true',
                        help="Whether we're test with a sentence_in_context data")
    parser.add_argument("--seq_start",
                        default=0,
                        type=int,
                        help="Window Bert start position for processing context data")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument('--log_file',
                        default=None,
                        type=str,
                        required=True,
                        help="log file")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=args.log_file,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    if args.only_test is False:
        train(args)

    test(args)

    if args.do_save is False:
        os.remove(os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s total seconds ---" % (time.time() - start_time))
