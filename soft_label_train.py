import json
import os
import re
import sys
import time
import ray
import math
import copy
import pickle
import random
import argparse
import collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import sent_tokenize
from nltk.translate import bleu_score
from metric import compute_bleu


# Start Ray for multi-processing
cpu_nums = 70
ray.init(num_cpus=cpu_nums)
# ray.init()


@ray.remote
def ray_compute_soft_label(thread_index, df_train_slicing_data, id_to_sent, dataset_name, print_freq=100):
    """ [TRAIN] Compute a series of bleu scores for every candidate sentence with respect to the reference sentences
    """
    # iterate through all the review data chunk
    cnt_row = 0
    cnt_sentence = 0
    sf = bleu_score.SmoothingFunction()     # smooth function of bleu score
    for idx, row_data in df_train_slicing_data.iterrows():
        user_id = row_data['user_id']
        item_id = row_data['item_id']
        cdd_sent_ids = row_data['candidate']        # a list of sent ids --> candidate sentences
        review_sent_ids = row_data['review']        # a list of sent ids --> review sentences
        assert isinstance(user_id, str)
        assert isinstance(item_id, str)
        assert isinstance(cdd_sent_ids, list)
        assert isinstance(review_sent_ids, list)
        assert isinstance(cdd_sent_ids[0], str)
        assert isinstance(review_sent_ids[0], str)
        # construct score dict
        score_dict = dict()                         # {sent_id: [bleu-1, bleu-2, bleu-3, bleu-4, google-bleu]}
        # NOTE: should we also include GLEU (https://www.nltk.org/_modules/nltk/translate/gleu_score.html)?
        # mapping sentence ids to sentence text
        # reference sentences are sentences in the gold-review
        # refs_sent = [id_to_sent[this_id] for this_id in review_sent_ids]
        refs_sent_tokens = [id_to_sent[this_id].split() for this_id in review_sent_ids]
        # loop through candidate sentences
        for this_cdd_sent_id in cdd_sent_ids:
            # hys_sent = id_to_sent[this_cdd_sent_id]
            hys_sent_tokens = id_to_sent[this_cdd_sent_id].split()
            cnt_sentence += 1
            # compute bleu-1 to bleu-4 and google bleu
            bleu_score_list = []
            # Bleu-1
            bleu_score_list.append(
                bleu_score.sentence_bleu(
                    refs_sent_tokens, hys_sent_tokens, smoothing_function=sf.method1, weights=[1.0, 0.0, 0.0, 0.0])
            )
            # Bleu-2
            bleu_score_list.append(
                bleu_score.sentence_bleu(
                    refs_sent_tokens, hys_sent_tokens, smoothing_function=sf.method1, weights=[0.5, 0.5, 0.0, 0.0])
            )
            # Bleu-3
            bleu_score_list.append(
                bleu_score.sentence_bleu(
                    refs_sent_tokens, hys_sent_tokens, smoothing_function=sf.method1, weights=[1.0/3, 1.0/3, 1.0/3, 0.0])
            )
            # Bleu-4
            bleu_score_list.append(
                bleu_score.sentence_bleu(
                    refs_sent_tokens, hys_sent_tokens, smoothing_function=sf.method1, weights=[0.25, 0.25, 0.25, 0.25])
            )
            # google-bleu
            bleu_score_list.append(
                compute_bleu([refs_sent_tokens], [hys_sent_tokens])
            )
            assert this_cdd_sent_id not in score_dict
            score_dict[this_cdd_sent_id] = bleu_score_list
        assert len(score_dict) == len(cdd_sent_ids)
        # write this into file
        this_dict = {
            'user_id': user_id,
            'item_id': item_id,
            'candidate': cdd_sent_ids,
            'review': review_sent_ids,
            'scores': score_dict
        }
        with open('./soft_label/BPR/{0}/{1}.json'.format(dataset_name, thread_index), 'a') as f_write:
            json.dump(this_dict, f_write)
            f_write.write("\n")
        cnt_row += 1
        if cnt_row % print_freq == 0:
            print("[Thread {0}] Finished {1} review instances".format(thread_index, cnt_row))
    print("[Thread {0}] Finished! Totally {1} lines.".format(thread_index, cnt_row))
    return cnt_row, cnt_sentence


if __name__ == "__main__":
    # Add argparser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='small_30', help='which dataset are we working on')
    parser.add_argument('--subset', type=str, default='valid', help='valid or test subset')     # only useful when calculating the proxy
    parser.add_argument('--window_size', type=int, default=1000, help='size of window')
    parser.add_argument('--threads', type=int, default=40, help='number of threads')
    parser.add_argument('--printfreq', type=int, default=100, help='frequency of print number of lines logging')
    args = parser.parse_args()
    # Read data
    # 1. Load Train Set and Test Set
    dir_path = '../Dataset/{}/'.format(args.dataset)
    # Load useritem to sentence ids mapping (compute soft label on train set)
    trainset_useritem_cdd_file = os.path.join(dir_path, 'train/useritem2sentids_multilines.json')
    train_review = []
    with open(trainset_useritem_cdd_file, 'r') as f:
        print("Load file: {}".format(trainset_useritem_cdd_file))
        for line in f:
            line_data = json.loads(line)
            user_id = line_data['user_id']
            item_id = line_data['item_id']
            cdd_sent_ids = line_data['candidate']
            review_sent_ids = line_data['review']
            train_review.append([user_id, item_id, cdd_sent_ids, review_sent_ids])
    print("Totally {0} lines of data in train dataset of {1}".format(len(train_review), args.dataset))
    # convert train review data into pandas dataframe
    df_train_data = pd.DataFrame(train_review, columns=['user_id', 'item_id', 'candidate', 'review'])
    assert len(df_train_data) == len(train_review)
    # 2. Load Sentence2ID and ID2Sentence Mapping
    # Load sentence2id mapping
    sentence2id_filepath = '../Dataset/{}/train/sentence/sentence2id.json'.format(args.dataset)
    with open(sentence2id_filepath, 'r') as f:
        print("Load file: {}".format(sentence2id_filepath))
        sent_to_id = json.load(f)
    # Load id2sentence mapping
    id2sentence_filepath = '../Dataset/{}/train/sentence/id2sentence.json'.format(args.dataset)
    with open(id2sentence_filepath, 'r') as f:
        print("Load file: {}".format(id2sentence_filepath))
        id_to_sent = json.load(f)
    # Load some hyperparameters
    window_size = args.window_size
    num_threads = args.threads
    data_instance_num = min(len(train_review), window_size*num_threads)
    print("We select {0} lines, with {1} threads and each thread has at most {2} lines".format(data_instance_num, num_threads, window_size))

    result_cnt = []
    # Construct and Allocate data slices
    for i in range(num_threads):
        start_row = i*window_size
        end_row = (i+1)*window_size
        result_cnt.append(
            ray_compute_soft_label.remote(
                i, df_train_data.iloc[start_row:end_row], id_to_sent, args.dataset, print_freq=args.printfreq
            )
        )
    ray_result_cnt = ray.get(result_cnt)
    total_num_UI = 0
    total_num_sent = 0
    for cnt_UI, cnt_sent in ray_result_cnt:
        total_num_UI += cnt_UI
        total_num_sent += cnt_sent
    print("Total number of user-item pair in the training set of {0}: {1}".format(args.dataset, total_num_UI))
    print("Total number of sentence in the training set of {0}: {1}".format(args.dataset, total_num_sent))
    ray.shutdown()
