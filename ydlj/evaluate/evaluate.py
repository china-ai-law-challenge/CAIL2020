import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

def normalize_answer(s):
    """
    Normalize a string.

    Args:
        s: (todo): write your description
    """

    def remove_articles(text):
        """
        Removes the articles from the text.

        Args:
            text: (str): write your description
        """
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        """
        Removes whitespace and remove leading whitespace.

        Args:
            text: (str): write your description
        """
        return ' '.join(text.split())

    def remove_punc(text):
        """
        Remove punctuation from string.

        Args:
            text: (str): write your description
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        """
        Convert text to lowercase

        Args:
            text: (str): write your description
        """
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Compute f1 score.

    Args:
        prediction: (array): write your description
        ground_truth: (array): write your description
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = list(normalized_prediction)
    ground_truth_tokens = list(normalized_ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    """
    Compute the answer.

    Args:
        prediction: (todo): write your description
        ground_truth: (todo): write your description
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    """
    Update the answer. f1 and gold.

    Args:
        metrics: (todo): write your description
        prediction: (todo): write your description
        gold: (todo): write your description
    """
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    """
    Update the metrics.

    Args:
        metrics: (dict): write your description
        prediction: (array): write your description
        gold: (array): write your description
    """
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    if fp + fn == 0:
        em, prec, recall, f1 = 1.0, 1.0, 1.0, 1.0
    else:
        em = 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    """
    Evaluate the bed file.

    Args:
        prediction_file: (str): write your description
        gold_file: (str): write your description
    """
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = str(dp['_id'])
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])

