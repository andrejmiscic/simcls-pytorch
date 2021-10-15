"""
    This script is used to reproduce the sentence-level evaluation of SimCLS, i.e. the bottom part of Table 3 and Fig. 3.
"""

import argparse
import os
import pickle
import sys

from typing import Dict, Tuple, List, Set, Any

import numpy as np
import pandas as pd
import plotnine as gg

from rouge_score.rouge_scorer import RougeScorer
from sklearn.utils import resample

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.plot_utils import theme_538


def match_all_summaries_to_doc(dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:

    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    matched_sentences_data = []

    for datapoint in dataset:
        doc_sentences = datapoint["doc"].split('\n')

        matched_sentences_data.append({
            "num_doc_sent": len(doc_sentences),
            "summary_sent_idx": __match_summary_doc_sentences(datapoint["summary"], doc_sentences, rouge_scorer),
            "origin_sent_idx": __match_summary_doc_sentences(datapoint["origin_candidate"], doc_sentences, rouge_scorer),
            "pred_sent_idx": __match_summary_doc_sentences(datapoint["predicted_candidate"], doc_sentences, rouge_scorer),
        })

    return matched_sentences_data


def __match_summary_doc_sentences(summary: str, doc_sentences: List[str], rouge_scorer: RougeScorer) -> Set[int]:
    summary_sentences = summary.split('\n')

    matched_indices = set()
    for sent in summary_sentences:
        rouge_scores = [rouge_scorer.score(doc_sent, sent) for doc_sent in doc_sentences]
        scores = [r["rouge1"].fmeasure + r["rouge2"].fmeasure + r["rougeLsum"].fmeasure for r in rouge_scores]

        matched_indices.add(np.argmax(scores))

    return matched_indices


def evaluate_sent_level_similarity(matched_sent_data: List[Dict[str, Any]]) -> None:

    tps_pred, fps_pred, fns_pred = [], [], []
    tps_orig, fps_orig, fns_orig = [], [], []

    summary_num_sents, pred_num_sents, orig_num_sents = [], [], []

    for datapoint in matched_sent_data:
        tp_pred, fp_pred, fn_pred = __evaluate_sent_perf(datapoint["summary_sent_idx"], datapoint["pred_sent_idx"])
        tp_orig, fp_orig, fn_orig = __evaluate_sent_perf(datapoint["summary_sent_idx"], datapoint["origin_sent_idx"])

        tps_pred.append(tp_pred)
        fps_pred.append(fp_pred)
        fns_pred.append(fn_pred)

        tps_orig.append(tp_orig)
        fps_orig.append(fp_orig)
        fns_orig.append(fn_orig)

        summary_num_sents.append(len(datapoint["summary_sent_idx"]))
        pred_num_sents.append(len(datapoint["pred_sent_idx"]))
        orig_num_sents.append(len(datapoint["origin_sent_idx"]))

    prec_orig_res, recall_orig_res, f1_orig_res = __calculate_performance_metrics(tps_orig, fps_orig, fns_orig)
    prec_pred_res, recall_pred_res, f1_pred_res = __calculate_performance_metrics(tps_pred, fps_pred, fns_pred)

    print(f"Summary num. sentences: {np.mean(summary_num_sents):.2f} +- {np.std(summary_num_sents):.2f}")

    print("Origin sent-level performance:")
    print(f"\t- precision: {prec_orig_res[0]:.4f}, [{prec_orig_res[1]:.4f}, {prec_orig_res[2]:.4f}]")
    print(f"\t- recall: {recall_orig_res[0]:.4f}, [{recall_orig_res[1]:.4f}, {recall_orig_res[2]:.4f}]")
    print(f"\t- F-1: {f1_orig_res[0]:.4f}, [{f1_orig_res[1]:.4f}, {f1_orig_res[2]:.4f}]")
    print(f"Origin num. sentences: {np.mean(orig_num_sents):.2f} +- {np.std(orig_num_sents):.2f}")

    print("SimCLS sent-level performance:")
    print(f"\t- precision: {prec_pred_res[0]:.4f}, [{prec_pred_res[1]:.4f}, {prec_pred_res[2]:.4f}]")
    print(f"\t- recall: {recall_pred_res[0]:.4f}, [{recall_pred_res[1]:.4f}, {recall_pred_res[2]:.4f}]")
    print(f"\t- F-1: {f1_pred_res[0]:.4f}, [{f1_pred_res[1]:.4f}, {f1_pred_res[2]:.4f}]")
    print(f"SimCLS num. sentences: {np.mean(pred_num_sents):.2f} +- {np.std(pred_num_sents):.2f}")


def __evaluate_sent_perf(summary_sent_idx: Set[int], sent_idx: Set[int]) -> Tuple[int, int, int]:
    tp = len(summary_sent_idx.intersection(sent_idx))
    fp = len(sent_idx - summary_sent_idx)
    fn = len(summary_sent_idx - sent_idx)

    return tp, fp, fn


def __calculate_performance_metrics(tps: List[int], fps: List[int], fns: List[int], iter_: int = 10000) -> \
        Tuple[Tuple[float, float, float], ...]:

    precision, recall, f1 = __calculate_metrics(np.sum(tps), np.sum(fps), np.sum(fns))

    precs, recalls, f1s = [], [], []
    for i in range(iter_):
        sample_tps, sample_fps, sample_fns = resample(tps, fps, fns, replace=True)
        sample_prec, sample_recall, sample_f1 = \
            __calculate_metrics(np.sum(sample_tps), np.sum(sample_fps), np.sum(sample_fns))
        precs.append(sample_prec)
        recalls.append(sample_recall)
        f1s.append(sample_f1)

    return (precision, np.percentile(precs, 2.5), np.percentile(precs, 97.5)), \
           (recall, np.percentile(recalls, 2.5), np.percentile(recalls, 97.5)), \
           (f1, np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))


def __calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f1 = 1.
    elif tp == 0 and (fp > 0 or fn > 0):
        precision = recall = f1 = 0.
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def plot_positional_bias(matched_sent_data: List[Dict[str, Any]], out_dir: str) -> None:

    summary_rel_positions, origin_rel_positions, pred_rel_positions = [], [], []

    for datapoint in matched_sent_data:
        summary_relative_pos = [round(idx / datapoint["num_doc_sent"], 1) for idx in list(datapoint["summary_sent_idx"])]
        summary_rel_positions.append(summary_relative_pos)

        origin_relative_pos = [round(idx / datapoint["num_doc_sent"], 1) for idx in list(datapoint["origin_sent_idx"])]
        origin_rel_positions.append(origin_relative_pos)

        pred_relative_pos = [round(idx / datapoint["num_doc_sent"], 1) for idx in list(datapoint["pred_sent_idx"])]
        pred_rel_positions.append(pred_relative_pos)

    bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1]
    summary_pos_hists = __calculate_position_bootstrap_ci(summary_rel_positions, bins)
    origin_pos_hists = __calculate_position_bootstrap_ci(origin_rel_positions, bins)
    pred_pos_hists = __calculate_position_bootstrap_ci(pred_rel_positions, bins)

    plot_df = pd.DataFrame({
        "x": bins[:-1] * 3,
        "y": summary_pos_hists[0] + origin_pos_hists[0] + pred_pos_hists[0],
        "lower": summary_pos_hists[1] + origin_pos_hists[1] + pred_pos_hists[1],
        "upper": summary_pos_hists[2] + origin_pos_hists[2] + pred_pos_hists[2],
        "Legend": ["Ref."] * len(bins[:-1]) + ["Origin"] * len(bins[:-1]) + ["SimCLS"] * len(bins[:-1])
    })

    plot_df["x"] *= 100
    plot = gg.ggplot(plot_df, gg.aes(x="x", y="y", fill="Legend")) + \
           gg.geom_bar(stat="identity", position="dodge", width=7.5) + \
           gg.geom_errorbar(gg.aes(ymin="lower", ymax="upper"), width=4., position=gg.position_dodge(width=7.5)) + \
           gg.scale_x_continuous(breaks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) + \
           gg.xlab("Relative position (%)") + gg.ylab("Ratio") + theme_538() + \
           gg.theme(text=gg.element_text(size=12), legend_position=(0.8, 0.6)) + \
           gg.guides(fill=gg.guide_legend(title="Legend:"))

    gg.ggsave(plot=plot, filename=os.path.join(out_dir, f"positional_bias.png"), dpi=600)


def __calculate_position_bootstrap_ci(positions: List[List[float]], bins: List[float], iter_: int = 1000) -> \
        Tuple[List[float], List[float], List[float]]:

    pos = __flatten(positions)
    hist, _ = np.histogram(pos, bins=bins)
    hist = hist / len(pos)

    hists = np.zeros((iter_, len(bins) - 1))

    for i in range(iter_):
        sample_pos = resample(positions, replace=True)
        pos = __flatten(sample_pos)
        sample_hist, _ = np.histogram(pos, bins=bins)
        hists[i, :] = sample_hist / len(pos)

    return hist.tolist(), np.percentile(hists, 2.5, axis=0).tolist(), np.percentile(hists, 97.5, axis=0).tolist()


def __flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentence-level eval. parser")

    # the pickle should be the output of build_experiment_data.py
    parser.add_argument("--data_path", help="Path to the pickle file.", type=str, required=True)
    parser.add_argument("--out_dir", help="Path to the directory to save resulting figures", type=str, required=True)

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()

    with open(args.data_path, "rb") as f:
        dataset = pickle.load(f)

    matched_sentences_data = match_all_summaries_to_doc(dataset)

    evaluate_sent_level_similarity(matched_sentences_data)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    plot_positional_bias(matched_sentences_data, args.out_dir)

