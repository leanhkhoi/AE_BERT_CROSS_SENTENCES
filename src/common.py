import json
import os
from collections import namedtuple, deque

import numpy as np
import torch
from torch.utils.data import RandomSampler, TensorDataset, DataLoader, SequentialSampler

Sentences = namedtuple('Sentences', [
    'orig_sentences', 'sentences', 'labels', 'lengths',
    'combined_sentences', 'combined_labels', 'sentence_numbers', 'sentence_starts'
])


def get_labels():
    return ["O", "B", "I"]


def read(inpput_file):
    with open(inpput_file) as f:
        return json.load(f)


class PreprocessConfig(object):

    def __init__(self, tokenizer, seq_len, no_context=False, seq_start=0):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.no_context = no_context
        self.seq_start = seq_start


class LoaderConfig(object):

    def __init__(self, batch_size=4, sampler=SequentialSampler):
        self.batch_size = batch_size
        self.sampler = sampler


def read_process(input_location, config: PreprocessConfig):
    lines = read(input_location)
    orig_sentences = []
    orig_labels = []
    for (i, ids) in enumerate(lines):
        orig_sentences.append(lines[ids]['sentence'])
        orig_labels.append(lines[ids]['label'])

    if config.no_context:
        return process_no_context(orig_sentences, orig_labels, config.tokenizer, config.seq_len)
    else:
        return process_sentences(orig_sentences, orig_labels, config.tokenizer, config.seq_len,
                                 config.seq_start)


# def convert_to_bert_features(sentences, labels, tag_map, tokenizer, seq_len):
#     tids = []
#     sids = []
#     masks = []
#     lids = []
#     for sentence, label in zip(sentences, labels):
#         tokens = ["[CLS]"] + sentence + ["[SEP]"]
#         token_ids = tokenizer.convert_tokens_to_ids(tokens)
#         segment_ids = [0] * len(token_ids)
#         mask = [1] * len(token_ids)
#         lb = [-1] + [tag_map[i] for i in label] + [-1]
#         if len(token_ids) < seq_len:
#             pad_len = seq_len - len(token_ids)
#             token_ids += tokenizer.convert_tokens_to_ids(["[PAD]"]) * pad_len
#             segment_ids += [0] * pad_len
#             mask += [0] * pad_len
#             lb += [-1] * pad_len
#         tids.append(token_ids)
#         sids.append(segment_ids)
#         masks.append(mask)
#         lids.append(lb)
#     return tids, sids, masks, lids

def to_data_loader(input_ids, segment_ids, masks, label_ids,
                   batch_size, sampler=SequentialSampler):
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_input_mask = torch.tensor(masks, dtype=torch.long)
    all_label_ids = torch.tensor(label_ids, dtype=torch.long)

    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    sampler = sampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)


def convert_to_features(sentences, labels, tag_map, tokenizer, seq_len):
    tids = []
    sids = []
    masks = []
    lids = []
    for sentence, label in zip(sentences, labels):
        tokens = ["[CLS]"] + sentence
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        mask = [1] * len(token_ids)
        lb = [0] + [tag_map[i] for i in label]
        if len(token_ids) < seq_len:
            pad_len = seq_len - len(token_ids)
            token_ids += tokenizer.convert_tokens_to_ids(["[PAD]"]) * pad_len
            segment_ids += [0] * pad_len
            mask += [0] * pad_len
            lb += [0] * pad_len
        tids.append(token_ids)
        sids.append(segment_ids)
        masks.append(mask)
        lids.append(lb)
    return tids, sids, masks, lids


def process_sentences(orig_sentences, orig_labels, tokenizer, max_seq_len, seq_start=0):
    # Tokenize words, split sentences to max_seq_len, and keep length
    # of each source word in tokens
    sentences, labels, lengths = tokenize_and_split_sentences(
        orig_sentences, orig_labels, tokenizer, max_seq_len)

    # Extend each sentence to include context sentences
    combined_sentences, combined_labels, sentence_numbers, sentence_starts = combine_sentences2(
        sentences, labels, max_seq_len - 1, seq_start)

    return Sentences(
        orig_sentences, sentences, labels, lengths, combined_sentences, combined_labels, sentence_numbers,
        sentence_starts)


def process_no_context(train_words, train_tags, tokenizer, seq_len):
    train_tokens, train_labels, train_lengths = tokenize_and_split_sentences(train_words, train_tags, tokenizer,
                                                                             seq_len)
    sentence_numbers = []
    sentence_starts = []
    for i, line in enumerate(train_tokens):
        sentence_numbers.append([i])
        sentence_starts.append([0])
    return Sentences(train_words, train_tokens, train_labels, train_lengths, train_tokens, train_labels,
                     sentence_numbers, sentence_starts)


def tokenize_and_split_sentences(orig_sentences, orig_labels, tokenizer, max_length):
    words, labels, lengths = [], [], []
    for s, l in zip(orig_sentences, orig_labels):
        split_s, split_l, lens = tokenize_and_split(s, l, tokenizer, max_length - 2)
        words.extend(split_s)
        labels.extend(split_l)
        lengths.extend(lens)
    return words, labels, lengths


def tokenize_and_split(sentence, word_labels, tokenizer, max_length):
    # Tokenize each word in sentence, propagate labels
    tokens, labels, lengths = [], [], []
    for word, label in zip(sentence, word_labels):
        tokenized = tokenizer.tokenize(word.lower())
        tokens.extend(tokenized)
        lengths.append(len(tokenized))
        for i, token in enumerate(tokenized):
            if i == 0:
                labels.append(label)
            else:
                if label == 'B':
                    labels.append('I')
                else:
                    labels.append(label)

    # Split into multiple sentences if too long
    split_tokens, split_labels = [], []
    start, end = 0, max_length
    while end < len(tokens):
        # Avoid splitting inside tokenized word
        while end > start and tokens[end].startswith('##'):
            end -= 1
        if end == start:
            end = start + max_length  # only continuations
        split_tokens.append(tokens[start:end])
        split_labels.append(labels[start:end])
        start = end
        end += max_length
    split_tokens.append(tokens[start:])
    split_labels.append(labels[start:])

    return split_tokens, split_labels, lengths


def combine_sentences2(lines, tags, max_seq, start=0):
    lines_in_sample = []
    linestarts_in_sample = []
    new_lines = []
    new_tags = []
    position = start

    for i, line in enumerate(lines):
        line_starts = []
        line_numbers = []
        if max_seq >= start + (len(line) + 1):  # 1 corresponding to [SEP] word
            new_line = [0] * start
            new_tag = [0] * start
            new_line.extend(line)
            new_tag.extend(tags[i])
            line_starts.append(start)
            line_numbers.append(i)
        else:
            position = max_seq - (len(line) + 1)  # 1 corresponding to [SEP] word
            new_line = [0] * position
            new_tag = [0] * position
            new_line.extend(line)
            new_tag.extend(tags[i])
            line_starts.append(position)
            line_numbers.append(i)
        j = 1
        next_idx = (i + j) % len(lines)
        ready = False
        while not ready:
            if max_seq >= (len(lines[next_idx]) + 1) + (len(new_line) + 1):  # 1 corresponding to [SEP] word
                new_line.append('[SEP]')
                new_tag.append('O')
                position = len(new_line)
                new_line.extend(lines[next_idx])
                new_tag.extend(tags[next_idx])
                line_starts.append(position)
                line_numbers.append(next_idx)
                j += 1
                next_idx = (i + j) % len(lines)
            else:
                new_line.append('[SEP]')
                new_tag.append('O')
                position = len(new_line)
                new_line.extend(lines[next_idx][0:(max_seq - position)])
                new_tag.extend(tags[next_idx][0:(max_seq - position)])
                ready = True

        # lines_in_sample.append(line_numbers)

        j = 1
        ready = False
        while not ready:
            counter = line_starts[0]
            # print(counter)
            prev_line = lines[i - j][:]
            prev_tags = tags[i - j][:]
            prev_line.append('[SEP]')
            prev_tags.append('O')
            # print(len(prev_line), len(prev_tags))
            if len(prev_line) <= counter:
                new_line[(counter - len(prev_line)):counter] = prev_line
                new_tag[(counter - len(prev_line)):counter] = prev_tags
                line_starts.insert(0, counter - len(prev_line))
                line_numbers.insert(0, i - j)  # negative numbers are indices to end of lines array
                j += 1
            else:
                if counter > 2:
                    new_line[0:counter] = prev_line[-counter:]
                    new_tag[0:counter] = prev_tags[-counter:]
                    ready = True
                else:
                    new_line[0:counter] = ['[PAD]'] * counter
                    new_tag[0:counter] = ['O'] * counter
                    ready = True
        new_lines.append(new_line)
        new_tags.append(new_tag)
        lines_in_sample.append(line_numbers)
        linestarts_in_sample.append(line_starts)
    return new_lines, new_tags, lines_in_sample, linestarts_in_sample


def get_predictions(predicted, sentences, combined_sentence_traces):
    first_pred = []
    final_pred = []
    predictions = [[] for _ in range(len(sentences))]
    for i, sample in enumerate(predicted):
        idx = 1
        for j, sentence_pos in enumerate(combined_sentence_traces[i]):
            predictions[sentence_pos].append(sample[idx:idx + len(sentences[sentence_pos])])
            if j == 0:
                first_pred.append(sample[idx:idx + len(sentences[sentence_pos])].tolist())
            idx += len(sentences[sentence_pos]) + 1
    for i, prediction in enumerate(predictions):
        pred = []
        arr = np.stack(prediction, axis=0)
        for j in arr.T:
            u, c = np.unique(j, return_counts=True)
            pred.append(int(u[np.argmax(c)]))
        final_pred.append(pred)
    return final_pred, first_pred


def get_predictions2(probs, sentences, combined_sentence_traces):
    first_pred = []
    final_pred = []
    predictions = []
    p_first = []
    for i, line in enumerate(sentences):
        predictions.append(np.zeros((len(line), probs.shape[-1])))  # create empty array for each line

    for i, sample in enumerate(probs):
        idx = 1
        for j, sentence_pos in enumerate(combined_sentence_traces[i]):
            if j == 0:
                p_first.append(sample[idx:idx + len(sentences[sentence_pos]), :])
            predictions[sentence_pos] += sample[idx:idx + len(sentences[sentence_pos]), :]
            idx += len(sentences[sentence_pos]) + 1

    for k, line in enumerate(predictions):
        final_pred.append(np.argmax(line, axis=-1).tolist())
        first_pred.append(np.argmax(p_first[k], axis=-1).tolist())

    return final_pred, first_pred


def write_result(fname, orig_sentences, lengths,
                 sentences, labels, predictions):
    lines = []
    with open(fname, 'w+') as f:
        toks = deque([val for sublist in sentences for val in sublist])
        labs = deque([val for sublist in labels for val in sublist])
        pred = deque([val for sublist in predictions for val in sublist])
        lengths = deque(lengths)
        raw_X = []
        y_pred = []
        for sentence in orig_sentences:
            raw_X.append(sentence)
            word_preds = []
            for word in sentence:
                label = labs.popleft()
                predicted = pred.popleft()
                word_preds.append(predicted)
                for i in range(int(lengths.popleft()) - 1):
                    labs.popleft()
                    pred.popleft()
                line = "{}\t{}\t{}\n".format(word, label, predicted)
                lines.append(line)
            y_pred.append(word_preds)

        json.dump({"raw_X": raw_X, "y_pred": y_pred}, f)
    f.close()
    return lines, sentences


def read_preprocess_load(input_location, preprocess_config: PreprocessConfig, loader_config: LoaderConfig):
    preprocessed_data = read_process(input_location, preprocess_config)

    combined_sentences = preprocessed_data.combined_sentences
    combined_labels = preprocessed_data.combined_labels
    tag_map = {l: i for i, l in enumerate(get_labels())}
    tokenizer = preprocess_config.tokenizer
    seq_len = preprocess_config.seq_len

    # vectorization
    input_ids, segment_ids, masks, label_ids = convert_to_features(combined_sentences, combined_labels, tag_map,
                                                                   tokenizer, seq_len)

    # transform to data loader
    data_loader = to_data_loader(input_ids, segment_ids, masks, label_ids, loader_config.batch_size,
                                 loader_config.sampler)
    return data_loader, preprocessed_data


def predict(model, data_loader, device):
    probs = []
    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy().tolist()

        probs.extend(logits)
    return probs
