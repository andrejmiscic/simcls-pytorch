import argparse
import os
import pickle
import random
import sys

from enum import Enum
from typing import List, Any, Dict, Tuple

import numpy as np

from bert_score import BERTScorer
from rouge_score.rouge_scorer import RougeScorer

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from evaluator import get_bootstrap_ci


class BaselineType(Enum):
    Origin = 0
    Min = 1
    Max = 2
    Random = 3


def evaluate_candidates(dataset: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[List[Dict[str, float]]]]:

    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    scores, rouge_scores = [], []
    for datapoint in dataset:
        summary = datapoint["gt_summary"]
        # reordering is necessary to recover the outputted order of generated summary. Without this, they are ordered by
        # their rouge score, which prevents correct evaluation.
        original_order = np.argsort(datapoint["candidate_order"])
        candidates = [datapoint["candidates"][o] for o in original_order]

        # evaluate all candidates based on their Rouge scores
        datapoint_scores, datapoint_rouges = [], []
        for candidate in candidates:
            rouge = rouge_scorer.score(summary, candidate)
            datapoint_rouges.append({"rouge1": rouge["rouge1"].fmeasure, "rouge2": rouge["rouge2"].fmeasure,
                                     "rougeL": rouge["rougeLsum"].fmeasure})
            # we select min/max candidates based on their average Rouge-1, -2, -L score
            datapoint_scores.append(rouge["rouge1"].fmeasure + rouge["rouge2"].fmeasure + rouge["rougeLsum"].fmeasure)

        scores.append(datapoint_scores)
        rouge_scores.append(datapoint_rouges)

    return scores, rouge_scores


def get_candidate_index(scores: List[float], baseline_type: BaselineType, rand: random.Random) -> int:
    if baseline_type == BaselineType.Origin.Origin:
        return 0
    elif baseline_type == BaselineType.Min:
        return np.argmin(scores)
    elif baseline_type == BaselineType.Max:
        return np.argmax(scores)
    elif baseline_type == BaselineType.Random:
        return rand.randint(0, len(scores) - 1)
    else:
        raise NotImplementedError("Provided baseline type not supported!")


def evaluate_baseline(dataset: List[Dict[str, Any]], scores: List[List[float]],
                      rouge_scores: List[List[Dict[str, float]]], baseline_type: BaselineType) -> None:

    rand = random.Random(0)
    predicted_candidates = []

    r1, r2, rl = [], [], []
    for i, datapoint in enumerate(dataset):
        cand_idx = get_candidate_index(scores[i], baseline_type, rand)

        predicted_candidates.append(datapoint["candidates"][cand_idx])
        r1.append(rouge_scores[i][cand_idx]["rouge1"])
        r2.append(rouge_scores[i][cand_idx]["rouge2"])
        rl.append(rouge_scores[i][cand_idx]["rougeL"])

    summaries = [d["gt_summary"] for d in dataset]

    _, _, bert_f_scores = BERTScorer(lang="en", rescale_with_baseline=True).score(predicted_candidates, summaries)
    bert_f_scores = bert_f_scores.cpu().numpy()

    for metric, res in zip(["Rouge-1", "Rouge-2", "Rouge-L", "BertScore Scaled"], [r1, r2, rl, bert_f_scores]):
        score, lower, upper = get_bootstrap_ci(res)
        print(f"{metric}: {score:.4f}, [{lower:.4f}, {upper:.4f}]")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline evaluation parser")
    parser.add_argument("--data_path", help="Path to the test pickle file", type=str, required=True)

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    scores, rouge_scores = evaluate_candidates(data)

    print("BASELINES RESULTS:")
    print("Origin:")
    evaluate_baseline(data, scores, rouge_scores, BaselineType.Origin)
    print("Min:")
    evaluate_baseline(data, scores, rouge_scores, BaselineType.Min)
    print("Max:")
    evaluate_baseline(data, scores, rouge_scores, BaselineType.Max)
    print("Random:")
    evaluate_baseline(data, scores, rouge_scores, BaselineType.Random)
