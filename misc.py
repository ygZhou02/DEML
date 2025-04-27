import nltk
from rouge_score import rouge_scorer
import pickle
import os
from transformers import AutoTokenizer
import json


def get_p_r_f1(rc_text_token: list, gt_text_token: list):
    total_pre = len(gt_text_token)
    total_rc = len(rc_text_token)
    precision = 0
    for item in rc_text_token:
        if item in gt_text_token:
            precision += 1
    precision = precision / total_pre
    recall = 0
    for item in gt_text_token:
        if item in rc_text_token:
            recall += 1
    recall = recall / total_rc
    if recall == 0 or precision == 0:
        f1 = 0
    else:
        f1 = 2 / (1 / recall + 1 / precision)
    ret = {"precision": precision, "recall": recall, "F1 score": f1}
    return ret


def get_rouge_score(rc_text_token: list, gt_text_token: list):
    rc_text, gt_text = " ".join(rc_text_token), " ".join(gt_text_token)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(rc_text, gt_text)
    rouge_ret = {}
    for key in scores:
        rouge_ret[key] = scores[key].fmeasure
    return rouge_ret


def get_bleu_score(rc_text_token: list, gt_text_token: list):
    BLEU1score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token, weights=([1]))
    BLEU2score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token, weights=(0.5, 0.5))
    BLEU4score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token,
                                                         weights=(0.25, 0.25, 0.25, 0.25))
    bleu_ret = {"bleu1": BLEU1score, "bleu2": BLEU2score, "bleu4": BLEU4score}
    return bleu_ret


def get_edit_distance(rc_text_token: list, gt_text_token: list):
    len_rc = len(rc_text_token)
    len_gt = len(gt_text_token)
    d = [[0] * (len_gt + 1) for i in range(len_rc + 1)]
    for i in range(1, len_rc + 1):
        d[i][0] = i
    for j in range(1, len_gt + 1):
        d[0][j] = j
    for j in range(1, len_gt + 1):
        for i in range(1, len_rc + 1):
            cost = 0 if rc_text_token[i - 1] == gt_text_token[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return {"edit distance": d[len_rc][len_gt]}


class RecoverMetric:
    def __init__(self, keywords=True, keywords_file="data/airline_keywords.json"):
        if keywords:
            if not os.path.exists(keywords_file):
                raise FileNotFoundError("keyword file {} does not exist!".format(keywords_file))
            with open(keywords_file, "r") as f:
                self.keywords = json.load(f)

    def get_metric(self, rc_text_token: list, gt_text_token: list):
        metrics = {}
        metrics.update(get_p_r_f1(rc_text_token, gt_text_token))
        metrics.update(get_rouge_score(rc_text_token, gt_text_token))
        metrics.update(get_bleu_score(rc_text_token, gt_text_token))
        metrics.update(get_edit_distance(rc_text_token, gt_text_token))
        return metrics

    def get_nerr(self, rc_text: str, gt_text: str):
        cnt = 0
        total = 0
        print(rc_text)
        print(gt_text)
        for keyword in self.keywords[gt_text]:
            total += 1
            print(keyword)
            if keyword in rc_text:
                cnt += 1
        if total == 0:   # no keywords
            return 1.0
        print(cnt/total)
        return cnt / total


path_to_tokenizer = "YOUR_TOKENIZER_PATH"
metrics = RecoverMetric(keywords_file="YOUR_KEYWORDS_FILE")
tokenizer = AutoTokenizer.from_pretrained(
    path_to_tokenizer,
    trust_remote_code=True,
    use_fast=False
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
pickle_fp = "YOUR_RESULT_PATH"

pr = rec = F1 = r1 = r2 = rL = b1 = b2 = b4 = e = e_l = nerr = 0
len_list = []
cnt = 0
for root, dirs, files in os.walk(pickle_fp):
    files.sort()
    print("num of samples: ", len(files))
    for i, file in enumerate(files):
        with open(os.path.join(pickle_fp, file), 'rb') as f:
            or_text, rc_text, tensor = pickle.load(f)
        rc_token = tokenizer.tokenize(rc_text)
        gt_token = tokenizer.tokenize(or_text)
        m = metrics.get_metric(rc_token, gt_token)
        pr += m['precision']
        rec += m['recall']
        F1 += m['F1 score']
        r1 += m['rouge1']
        r2 += m['rouge2']
        rL += m['rougeL']
        b1 += m['bleu1']
        b2 += m['bleu2']
        b4 += m['bleu4']
        e += m['edit distance']
        len_list.append(len(gt_token))
        e_l += (m['edit distance']/len(gt_token))
        nerr += metrics.get_nerr(rc_text, or_text)
        cnt += 1
print('precision', pr / cnt)
print('recall', rec / cnt)
print('F1 score', F1 / cnt)
print('rouge1', r1 / cnt)
print('rouge2', r2 / cnt)
print('rougeL', rL / cnt)
print('bleu1', b1 / cnt)
print('bleu2', b2 / cnt)
print('bleu4', b4 / cnt)
print("edit distance", e / cnt)
print("e / l", e_l / cnt)
print("nerr", nerr / cnt)
print(len_list)