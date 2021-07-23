import numpy as np
import nltk
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
import math
from collections import Counter


def get_feature_recall_precision_rouge(pred, ref):
    ''' using rouge-score to compute feature precision/recall
    :param pred: list of features which appears in the predicted sentences
    :param ref: list of features that are in the reference sentences

    :return recall: the recall score of the pred features
    :return precision: the precision score of the pred features
    '''
    pred_concat = " ".join(pred)
    ref_concat = " ".join(ref)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(ref_concat, pred_concat)
    return scores['rouge1'].recall, scores['rouge1'].precision, scores['rouge1'].fmeasure


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            # print(ngram)
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, hypothese_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        hypothese_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        BLEU score
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    reference_length = 0
    hypothesis_length = 0

    for (references, hypothesis) in zip(reference_corpus, hypothese_corpus):
        reference_length += min(len(r) for r in references)
        hypothesis_length += len(hypothesis)

        merged_ref_ngram_counts = Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

        hyp_ngram_counts = _get_ngrams(hypothesis, max_order)
        overlap = hyp_ngram_counts & merged_ref_ngram_counts

        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]

        for order in range(1, max_order+1):
            possible_matches = len(hypothesis)-order+1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i]+1.0)/(possible_matches_by_order[i]+1.0))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i])/possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0/max_order)*math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(hypothesis_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1-1.0/ratio)

    bleu = geo_mean*bp

    return bleu


def get_sentence_bleu(references, hypotheses, types=[1, 2, 3, 4]):
    """ This is used to compute sentence-level bleu
    param: references: list of reference sentences, each reference sentence is a list of tokens
    param: hypoyheses: hypotheses sentences, this is a list of tokenized tokens
    return:
        bleu-1, bleu-2, bleu-3, bleu-4
    """
    type_weights = [[1.0, 0., 0., 0],
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0/3, 1.0/3, 1.0/3, 0.0],
                    [0.25, 0.25, 0.25, 0.25]]
    sf = bleu_score.SmoothingFunction()
    bleu_1_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[0])
    bleu_2_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[1])
    bleu_3_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[2])
    bleu_4_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[3])
    return bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score


def get_bleu(references, hypotheses, types=[1, 2, 3, 4]):
    """ This is used to compute the corpus-level bleu
    """
    type_weights = [[1.0, 0., 0., 0],
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0/3, 1.0/3, 1.0/3, 0.0],
                    [0.25, 0.25, 0.25, 0.25]]

    totals = [0.0] * len(types)

    sf = bleu_score.SmoothingFunction()

    num = 0

    for (reference, hypothesis) in zip(references, hypotheses):

        for j, t in enumerate(types):
            weights = type_weights[t-1]
            totals[j] += bleu_score.sentence_bleu(reference, hypothesis, smoothing_function=sf.method1, weights=weights)

        num += 1.0

    totals = [total/num for total in totals]

    return totals
