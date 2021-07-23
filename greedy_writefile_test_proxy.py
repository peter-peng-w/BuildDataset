import json
import os
import sys
import time
import ray
from rouge import Rouge
from rouge_score import rouge_scorer
import numpy as np
import copy
import pickle
import random
import argparse


# Start Ray for multi-processing
cpu_nums = 70
ray.init(num_cpus=cpu_nums)
# ray.init()


# Compute the rouge score for a hyps and a ground truth
def rouge_eval_prev(hyps, ref):
    rouge = Rouge()
    try:
        score = rouge.get_scores(hyps, ref)[0]
        mean_score = np.mean([score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']])
        rouge_1_f_score = score['rouge-1']['f']
        rouge_2_f_score = score['rouge-2']['f']
        rouge_l_f_score = score['rouge-l']['f']
    except:
        mean_score = 0.0
        rouge_1_f_score = 0.0
        rouge_2_f_score = 0.0
        rouge_l_f_score = 0.0
    return mean_score, rouge_1_f_score, rouge_2_f_score, rouge_l_f_score


def rouge_eval(hyps, ref):
    # This is the implementation given by Google
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # build a result dict
    result_score = dict()
    result_score['rouge1'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    result_score['rouge2'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    result_score['rougeL'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}

    try:
        scores = scorer.score(ref, hyps)
        rouge_1_f_score = scores['rouge1'].fmeasure
        rouge_2_f_score = scores['rouge2'].fmeasure
        rouge_l_f_score = scores['rougeL'].fmeasure
        mean_score = np.mean([rouge_1_f_score, rouge_2_f_score, rouge_l_f_score])
        result_score['rouge1'] = {
            'precision': scores['rouge1'].precision,
            'recall': scores['rouge1'].recall,
            'f': scores['rouge1'].fmeasure
            }
        result_score['rouge2'] = {
            'precision': scores['rouge2'].precision,
            'recall': scores['rouge2'].recall,
            'f': scores['rouge2'].fmeasure
            }
        result_score['rougeL'] = {
            'precision': scores['rougeL'].precision,
            'recall': scores['rougeL'].recall,
            'f': scores['rougeL'].fmeasure
            }
    except:
        mean_score = 0.0
    return mean_score, result_score


def calLabel(article, abstract):
    """
    :param article: list of candidate sentences
    :param abstract: true review
    :return selected: a list of idx of the selected the sentences which can maximize the rouge score
    :return best_rouge: the best rouge score that this greedy algorithm can reach
    """
    # start_func = time.time()
    hyps_list = article
    refer = abstract
    scores = []
    result_rouge_scores = []
    selected_sent_cnt = 0
    result_score = dict()
    result_score['rouge1'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    result_score['rouge2'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    result_score['rougeL'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    # time_computing_rouge = 0.0
    # cnt_computing_rouge = 0

    # If the candidate article is 0, then return 0.0
    if len(hyps_list) == 0 or len(abstract) == 0:
        return [], result_score

    for hyps in hyps_list:
        # start = time.time()
        mean_score, result_scores = rouge_eval(hyps, refer)
        # end = time.time()
        # time_computing_rouge += (end-start)
        # cnt_computing_rouge += 1
        scores.append(mean_score)
        result_rouge_scores.append(result_scores)

    # A set of index that are selected as proxy sentences
    selected = set([int(np.argmax(scores))])
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    cur_selected_idx = list(selected)[0]
    best_result_rouge_score = result_rouge_scores[cur_selected_idx]
    best_hyps = hyps_list[cur_selected_idx]

    # # if the true review is empty, the best rouge score can only be 0.0
    # if best_rouge == 0.0:
    #     return selected, best_rouge

    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_result_score = dict()
        cur_max_result_score['rouge1'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
        cur_max_result_score['rouge2'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
        cur_max_result_score['rougeL'] = {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
        cur_max_hyps = ""
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.add(i)
                hyps = " ".join([hyps_list[idx] for idx in temp])
                # start = time.time()
                cur_rouge, cur_result_score = rouge_eval(hyps, refer)
                # end = time.time()
                # time_computing_rouge += (end-start)
                # cnt_computing_rouge += 1
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_result_score = cur_result_score
                    cur_max_hyps = hyps
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.add(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
            best_result_rouge_score = cur_max_result_score
            best_hyps = cur_max_hyps
        else:
            break
    # print(selected, best_rouge)
    # end_func = time.time()
    # print("Time used for this function: {0}, computing rouge: {1}".format(end_func - start_func, time_computing_rouge))
    # print("Average time used for a single rouge computing: {}".format(time_computing_rouge/cnt_computing_rouge))
    # print("Number of candidate sentences: {}".format(len(article)))
    return selected, best_rouge, best_result_rouge_score, best_hyps


@ray.remote
def ray_best_rouge(index, test_reviews, dataset_name, subset_name):
    """
    :param id_to_sent: dict which mapping id to sentence
    :param user_sent_ids: dict which mapping user to its relevant sents' ids
    :param item_sent_ids: dict which mapping item to its relevant sents' ids
    :param test_reviews: list of test reviews in a format of [[user_id, item_id, rating, review],...]
    :return : a list of best rouge score for each review
    """
    # Load some dictionaries
    train_id2sentence_filepath = '../Dataset/{}/train/sentence/id2sentence.json'.format(dataset_name)
    with open(train_id2sentence_filepath, 'r') as f_train_sent:
        print("Load file: {}".format(train_id2sentence_filepath))
        id_to_sent_trainset = json.load(f_train_sent)
    test_id2sentence_filepath = '../Dataset/{}/test/sentence/id2sentence.json'.format(dataset_name)
    with open(test_id2sentence_filepath, 'r') as f_test_sent:
        print("Load file: {}".format(test_id2sentence_filepath))
        id_to_sent_testset = json.load(f_test_sent)

    best_rouge_scores = []
    best_rouge_1_scores = []
    best_rouge_2_scores = []
    best_rouge_l_scores = []
    best_rouge_1_precison = []
    best_rouge_1_recall = []
    best_rouge_2_precision = []
    best_rouge_2_recall = []
    best_rouge_l_precision = []
    best_rouge_l_recall = []
    cnt = 0
    hard_clip_size = 1500       # if the candidate set is too large, we clip it
    do_hard_clip = False
    for test_review_chunk in test_reviews:
        user_id = int(test_review_chunk[0])
        item_id = int(test_review_chunk[1])
        candidate_set = test_review_chunk[2]
        review_set = test_review_chunk[3]
        # If candidate set is too large, we sample a subset from the candidate set
        if do_hard_clip and len(candidate_set) > hard_clip_size:
            # cnt += 1
            # if cnt % 5 == 0:
            #     print("[Thread {0}] Finished {1} review instances".format(index, cnt))
            # continue
            candidate_set = random.sample(candidate_set, hard_clip_size)
        # Get the corresponding sentences
        this_candidate_sents = []
        for this_cand in candidate_set:
            this_candidate_sents.append(id_to_sent_trainset[this_cand])
        # Get the corresponding gold review
        this_review_sents = []
        for this_rev in review_set:
            this_review_sents.append(id_to_sent_testset[this_rev])
        true_review = " ".join(this_review_sents)
        # Compute the label and the best rouge score
        selected_sents, best_rouge, best_result_rouge_scores, best_hyps = calLabel(this_candidate_sents, true_review)
        # add rouge score statistics
        best_rouge_scores.append(best_rouge)
        best_rouge_1_scores.append(best_result_rouge_scores['rouge1']['f'])
        best_rouge_2_scores.append(best_result_rouge_scores['rouge2']['f'])
        best_rouge_l_scores.append(best_result_rouge_scores['rougeL']['f'])
        best_rouge_1_precison.append(best_result_rouge_scores['rouge1']['precision'])
        best_rouge_1_recall.append(best_result_rouge_scores['rouge1']['recall'])
        best_rouge_2_precision.append(best_result_rouge_scores['rouge2']['precision'])
        best_rouge_2_recall.append(best_result_rouge_scores['rouge2']['recall'])
        best_rouge_l_precision.append(best_result_rouge_scores['rougeL']['precision'])
        best_rouge_l_recall.append(best_result_rouge_scores['rougeL']['recall'])
        # convert selected_sents to sent_ids
        proxy_sent_ids = [candidate_set[idx] for idx in selected_sents]
        cnt += 1
        # write the selected sentence and gold review
        result = {
            "user": user_id, "item": item_id, "select_text": best_hyps, "review_text": true_review,
            "candidate_ids": candidate_set, "review_ids": review_set, "proxy_ids": proxy_sent_ids
            }
        with open("./proxy/{0}/{1}/{2}.json".format(dataset_name, subset_name, index), 'a') as f_write:
            json.dump(result, f_write)
            f_write.write("\n")

        if cnt % 100 == 0:
            print("[Thread {0}] Finished {1} review instances".format(index, cnt))
    print("[Thread {0}] Finished! Totally {1} lines.".format(index, cnt))
    return best_rouge_scores, best_rouge_1_scores, best_rouge_2_scores, best_rouge_l_scores, best_rouge_1_precison, best_rouge_1_recall, best_rouge_2_precision, best_rouge_2_recall, best_rouge_l_precision, best_rouge_l_recall


if __name__ == "__main__":
    # Add argparser
    parser = argparse.ArgumentParser(description='Set hyperparameters for computing proxy.')
    parser.add_argument('--dataset', type=str, default='small_30', help='which dataset are we working on')
    parser.add_argument('--subset', type=str, default='valid', help='valid or test subset')
    parser.add_argument('--window_size', type=int, default=100, help='size of window')
    parser.add_argument('--bias_size', type=int, default=0, help='size of the bias')
    parser.add_argument('--threads', type=int, default=40, help='number of threads')
    parser.add_argument('--bias_threads', type=int, default=0, help='number of bias of threads')
    args = parser.parse_args()
    # The main function
    # Load the test dataset
    test_review = []
    file_path_0 = '../Dataset/{0}/{1}/useritem2sentids_test_multilines.json'.format(args.dataset, args.subset)
    cnt = 0
    with open(file_path_0) as f_0:
        print("Load whole file: {}".format(file_path_0))
        for line in f_0:
            line_data = json.loads(line)
            user_id = line_data['user_id']
            item_id = line_data['item_id']
            candidate_set = line_data['candidate']      # sent_id from the training set, send_id is str
            review_set = line_data['review']            # sent_id from the testing set, send_id is str
            test_review.append([user_id, item_id, candidate_set, review_set])
            cnt += 1
    print("Working on dataset: {0} \t subset: {1}".format(args.dataset, args.subset))
    print("{} lines of test data loaded!".format(cnt))
    # Start parallel greedy label assigning and rouge score computing
    # NOTE: Since we are runing this on multiple machines, we can't shuffle the whole dataset
    # random.shuffle(test_review)
    results_rouge = []
    window_size = args.window_size
    num_threads = args.threads
    bias_size = args.bias_size
    data_instance_num = min(len(test_review[bias_size:(bias_size+window_size*num_threads)]), window_size*num_threads)
    print("We select {0} lines, with {1} threads and each thread has at most {2} lines".format(data_instance_num, num_threads, window_size))
    print("Get test data start from line {0} and ends at line {1}".format(bias_size, bias_size + data_instance_num - 1))
    # Construct and Allocate data slices
    for i in range(num_threads):
        results_rouge.append(ray_best_rouge.remote(
            (i+args.bias_threads), test_review[(bias_size+i*window_size):(bias_size+(i+1)*window_size)], args.dataset, args.subset)
        )
    res_best_rouge = ray.get(results_rouge)
    res_mean_rouge_score = []
    res_rouge_1_f_score = []
    res_rouge_2_f_score = []
    res_rouge_l_f_score = []
    res_rouge_1_precison = []
    res_rouge_1_recall = []
    res_rouge_2_precision = []
    res_rouge_2_recall = []
    res_rouge_l_precision = []
    res_rouge_l_recall = []
    for score in res_best_rouge:
        mean_rouge = score[0]
        rouge_1_fscore = score[1]
        rouge_2_fscore = score[2]
        rouge_l_fscore = score[3]
        rouge_1_precision = score[4]
        rouge_1_recall = score[5]
        rouge_2_precision = score[6]
        rouge_2_recall = score[7]
        rouge_l_precision = score[8]
        rouge_l_recall = score[9]
        res_mean_rouge_score.extend(mean_rouge)
        res_rouge_1_f_score.extend(rouge_1_fscore)
        res_rouge_2_f_score.extend(rouge_2_fscore)
        res_rouge_l_f_score.extend(rouge_l_fscore)
        res_rouge_1_precison.extend(rouge_1_precision)
        res_rouge_1_recall.extend(rouge_1_recall)
        res_rouge_2_precision.extend(rouge_2_precision)
        res_rouge_2_recall.extend(rouge_2_recall)
        res_rouge_l_precision.extend(rouge_l_precision)
        res_rouge_l_recall.extend(rouge_l_recall)
    # Print Results
    print("Totally lines of data: {}".format(len(res_mean_rouge_score)))
    print("Average best rouge score: {}".format(np.mean(res_mean_rouge_score)))
    print("Average rouge-1 f-score: {}".format(np.mean(res_rouge_1_f_score)))
    print("Average rouge-2 f-score: {}".format(np.mean(res_rouge_2_f_score)))
    print("Average rouge-l f-score: {}".format(np.mean(res_rouge_l_f_score)))
    print("Average rouge-1 precision: {}".format(np.mean(res_rouge_1_precison)))
    print("Average rouge-2 precision: {}".format(np.mean(res_rouge_2_precision)))
    print("Average rouge-l precision: {}".format(np.mean(res_rouge_l_precision)))
    print("Average rouge-1 recall: {}".format(np.mean(res_rouge_1_recall)))
    print("Average rouge-2 recall: {}".format(np.mean(res_rouge_2_recall)))
    print("Average rouge-l recall: {}".format(np.mean(res_rouge_l_recall)))
    # write rouge score statistics into file
    with open('./proxy/{0}/{1}/rouge_score_{2}.txt'.format(args.dataset, args.subset, args.bias_threads), 'w') as fw:
        fw.write("Totally lines of data: {}\n".format(len(res_mean_rouge_score)))
        fw.write("Average best rouge score: {}\n".format(np.mean(res_mean_rouge_score)))
        fw.write("Average rouge-1 f-score: {}\n".format(np.mean(res_rouge_1_f_score)))
        fw.write("Average rouge-2 f-score: {}\n".format(np.mean(res_rouge_2_f_score)))
        fw.write("Average rouge-l f-score: {}\n".format(np.mean(res_rouge_l_f_score)))
        fw.write("Average rouge-1 precision: {}\n".format(np.mean(res_rouge_1_precison)))
        fw.write("Average rouge-2 precision: {}\n".format(np.mean(res_rouge_2_precision)))
        fw.write("Average rouge-l precision: {}\n".format(np.mean(res_rouge_l_precision)))
        fw.write("Average rouge-1 recall: {}\n".format(np.mean(res_rouge_1_recall)))
        fw.write("Average rouge-2 recall: {}\n".format(np.mean(res_rouge_2_recall)))
        fw.write("Average rouge-l recall: {}\n".format(np.mean(res_rouge_l_recall)))
    ray.shutdown()
