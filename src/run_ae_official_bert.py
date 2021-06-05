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
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertPreTrainedModel, BertModel, \
    BertLayer as OfficialBertLayer, AdamW, get_linear_schedule_with_warmup
import absa_data_utils as data_utils
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
                loss = self.crf_layers[i](logits.view(100, -1, self.num_labels), labels.view(100, -1))
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
    processor = data_utils.AeProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, "ae")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # >>>>> validation
    if args.no_valid is False:
        valid_examples = processor.get_dev_examples(args.data_dir)
        valid_features = data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "ae")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                   valid_all_label_ids)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)

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
            # >>>> perform validation at the end of each epoch .
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


def test(args, dev_as_test=None, output_dir=None, model=None, model_dir=None):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)
    if output_dir is None:
        output_dir = args.output_dir
    if model_dir is None:
        model_dir = args.output_dir

    processor = data_utils.AeProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    if dev_as_test:
        data_dir = os.path.join(args.data_dir, 'dev_as_test')
    else:
        data_dir = args.data_dir
    eval_examples = processor.get_test_examples(data_dir)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer,
                                                            "ae")

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if model is None:
        _model = torch.load(os.path.join(model_dir, "model.pt"))
        _model.to(device)
        _model.eval()
    else:
        _model = model

    full_logits = []
    full_label_ids = []
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch

        with torch.no_grad():
            logits = _model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        full_logits.extend(logits.tolist())
        full_label_ids.extend(label_ids.tolist())

    output_eval_json = os.path.join(output_dir, "predictions.json")
    with open(output_eval_json, "w") as fw:
        assert len(full_logits) == len(eval_examples)
        # sort by original order for evaluation
        recs = {}
        for qx, ex in enumerate(eval_examples):
            recs[int(ex.guid.split("-")[1])] = {"sentence": ex.text_a, "idx_map": ex.idx_map,
                                                "logit": full_logits[qx][1:]}  # skip the [CLS] tag.
        full_logits = [recs[qx]["logit"] for qx in range(len(full_logits))]
        raw_X = [recs[qx]["sentence"] for qx in range(len(eval_examples))]
        idx_map = [recs[qx]["idx_map"] for qx in range(len(eval_examples))]
        json.dump({"logits": full_logits, "raw_X": raw_X, "idx_map": idx_map}, fw)


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
    main()
