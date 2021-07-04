import argparse
import os
import json
import time

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import read

nlp = spacy.load("en_core_web_sm")


def process(data_dir):
    # paths
    train_path = os.path.join(data_dir, "train.json")
    train_150_path = os.path.join(data_dir, "train-150.json")
    dev_path = os.path.join(data_dir, "dev.json")
    test_path = os.path.join(data_dir, "test.json")
    dev_as_test_path = os.path.join(data_dir, "dev_as_test", "test.json")

    # read files
    train_lines = read(train_path)
    train_150_lines = read(train_150_path)
    dev_lines = read(dev_path)
    test_lines = read(test_path)
    dev_as_test_lines = read(dev_as_test_path)

    # transform to docs (not tokens)
    train_docs = []
    train_150_docs = []
    dev_docs = []
    test_docs = []
    dev_as_test_docs = []

    for (i, ids) in enumerate(train_lines):
        train_docs.append(doc(train_lines[ids]['sentence']))

    for (i, ids) in enumerate(train_150_lines):
        train_150_docs.append(doc(train_150_lines[ids]['sentence']))

    for (i, ids) in enumerate(dev_lines):
        dev_docs.append(doc(dev_lines[ids]['sentence']))

    for (i, ids) in enumerate(test_lines):
        test_docs.append(doc(test_lines[ids]['sentence']))

    for (i, ids) in enumerate(dev_as_test_lines):
        dev_as_test_docs.append(doc(dev_as_test_lines[ids]['sentence']))

    # apply TF-IDF
    vectorizer = TfidfVectorizer(stop_words=['the', 'this', 'that', "it", "is", "but", "be", "a"], max_features=400)
    train_tfidf = vectorizer.fit_transform(train_docs)
    train_150_tfidf = vectorizer.fit_transform(train_150_docs)
    dev_tfidf = vectorizer.transform(dev_docs)
    test_tfidf = vectorizer.transform(test_docs)
    dev_as_test_tfidf = vectorizer.transform(dev_as_test_docs)

    # calculating similarity
    train_sim = cosine_similarity(train_tfidf, train_tfidf)
    train_sim_indexes = np.argsort(train_sim * -1)  # multiply -1 to sort similarities in descending order

    train_150_sim = cosine_similarity(train_150_tfidf, train_150_tfidf)
    train_150_sim_indexes = np.argsort(train_150_sim * -1)  # multiply -1 to sort similarities in descending order

    dev_sim = cosine_similarity(dev_tfidf, dev_tfidf)
    dev_sim_indexes = np.argsort(dev_sim * -1)  # multiply -1 to sort similarities in descending order

    test_sim = cosine_similarity(test_tfidf, test_tfidf)
    test_sim_indexes = np.argsort(test_sim * -1)  # multiply -1 to sort similarities in descending order

    dev_as_test_sim = cosine_similarity(dev_as_test_tfidf, dev_as_test_tfidf)
    dev_as_test_sim_indexes = np.argsort(dev_as_test_sim * -1)  # multiply -1 to sort similarities in descending order

    # save similarity matrix
    train_output = os.path.join(data_dir, "train-similarity.json")
    train_150_output = os.path.join(data_dir, "train-150-similarity.json")
    dev_output = os.path.join(data_dir, "dev-similarity.json")
    test_output = os.path.join(data_dir, "test-similarity.json")
    dev_as_test_output = os.path.join(data_dir, "dev_as_test", "test-similarity.json")
    with open(train_output, 'w+') as f:
        json.dump({"matrix": train_sim_indexes.tolist()}, f)
    f.close()
    with open(train_150_output, 'w+') as f:
        json.dump({"matrix": train_150_sim_indexes.tolist()}, f)
    f.close()
    with open(dev_output, 'w+') as f:
        json.dump({"matrix": dev_sim_indexes.tolist()}, f)
    f.close()
    with open(test_output, 'w+') as f:
        json.dump({"matrix": test_sim_indexes.tolist()}, f)
    f.close()
    with open(dev_as_test_output, 'w+') as f:
        json.dump({"matrix": dev_as_test_sim_indexes.tolist()}, f)
    f.close()

    # sorted_indexs = np.argsort(sim_matrix * -1)
    # sorted = np.sort(sim_matrix*-1)*-1
    #
    # # print(f"Top most similarity values of doc {0}: {sorted[0]}")
    # # print(f'Top most similarity indexs of doc {0}: {sorted_indexs[0]}')
    # doc_index = 0
    # print(f"Doc {doc_index}: {docs[doc_index]}")
    # docs = np.array(docs)
    # print(f"Similarity docs with doc {doc_index}: {docs[sorted_indexs[doc_index][1:6].tolist()]}")

    return train_sim, dev_sim, test_sim, dev_as_test_sim


def doc(tks):
    spaces = [True] * (len(tks) - 1) + [False]
    for i, token in enumerate(tks):
        if i != 0 and token == "n\'t":
            spaces[i - 1] = False

    return spacy.tokens.Doc(nlp.vocab, words=tks, spaces=spaces).text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir containing json files.")

    args = parser.parse_args()

    process(args.data_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s total seconds ---" % (time.time() - start_time))
