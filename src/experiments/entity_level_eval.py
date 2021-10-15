"""
    This script is used to reproduce the entity-level evaluation of SimCLS, i.e. the upper part of Table 3.s
"""

import argparse
import pickle

from typing import Dict, Tuple, List, Set

import numpy as np
import spacy

from sklearn.utils import resample


class EntityLevelEvaluator:

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def evaluate(self, dataset: List[Dict[str, str]]) -> None:

        tps_pred, fps_pred, fns_pred = [], [], []
        tps_orig, fps_orig, fns_orig = [], [], []

        for datapoint in dataset:
            doc_ents = self.__get_entities(datapoint["doc"])
            summary_ents = self.__get_entities(datapoint["summary"])
            origin_ents = self.__get_entities(datapoint["origin_candidate"])
            pred_ents = self.__get_entities(datapoint["predicted_candidate"])

            positives = summary_ents.intersection(doc_ents)
            tp_pred, fp_pred, fn_pred = self.__calculate_datapoint_performance(pred_ents, summary_ents, positives)
            tp_orig, fp_orig, fn_orig = self.__calculate_datapoint_performance(origin_ents, summary_ents, positives)

            tps_pred.append(tp_pred)
            fps_pred.append(fp_pred)
            fns_pred.append(fn_pred)

            tps_orig.append(tp_orig)
            fps_orig.append(fp_orig)
            fns_orig.append(fn_orig)

        prec_orig_res, recall_orig_res, f1_orig_res = self.__bootstrap_ci(tps_orig, fps_orig, fns_orig)
        prec_pred_res, recall_pred_res, f1_pred_res = self.__bootstrap_ci(tps_pred, fps_pred, fns_pred)

        print("Origin entity-level performance:")
        print(f"\t- precision: {prec_orig_res[0]:.4f}, [{prec_orig_res[1]:.4f}, {prec_orig_res[2]:.4f}]")
        print(f"\t- recall: {recall_orig_res[0]:.4f}, [{recall_orig_res[1]:.4f}, {recall_orig_res[2]:.4f}]")
        print(f"\t- F-1: {f1_orig_res[0]:.4f}, [{f1_orig_res[1]:.4f}, {f1_orig_res[2]:.4f}]")

        print("SimCLS entity-level performance:")
        print(f"\t- precision: {prec_pred_res[0]:.4f}, [{prec_pred_res[1]:.4f}, {prec_pred_res[2]:.4f}]")
        print(f"\t- recall: {recall_pred_res[0]:.4f}, [{recall_pred_res[1]:.4f}, {recall_pred_res[2]:.4f}]")
        print(f"\t- F-1: {f1_pred_res[0]:.4f}, [{f1_pred_res[1]:.4f}, {f1_pred_res[2]:.4f}]")

    def __get_entities(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        entities = set([ent.lemma_ for ent in doc.ents])

        return entities

    def __calculate_datapoint_performance(self, entities: Set[str], summary_ents: Set[str], positives: Set[str]) -> \
            Tuple[int, int, int]:

        tp = len(entities.intersection(positives))
        fp = len(entities - summary_ents)
        fn = len(summary_ents - entities)

        return tp, fp, fn

    def __bootstrap_ci(self, tps: List[int], fps: List[int], fns: List[int], iter_: int = 10000) -> \
            Tuple[Tuple[float, float, float], ...]:

        precision, recall, f1 = self.__calculate_metrics(np.sum(tps), np.sum(fps), np.sum(fns))

        precs, recalls, f1s = [], [], []
        for i in range(iter_):
            sample_tps, sample_fps, sample_fns = resample(tps, fps, fns, replace=True)
            sample_prec, sample_recall, sample_f1 = \
                self.__calculate_metrics(np.sum(sample_tps), np.sum(sample_fps), np.sum(sample_fns))
            precs.append(sample_prec)
            recalls.append(sample_recall)
            f1s.append(sample_f1)

        return (precision, np.percentile(precs, 2.5), np.percentile(precs, 97.5)), \
               (recall, np.percentile(recalls, 2.5), np.percentile(recalls, 97.5)), \
               (f1, np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))

    def __calculate_metrics(self, tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        if tp == 0 and fp == 0 and fn == 0:
            precision = recall = f1 = 1.
        elif tp == 0 and (fp > 0 or fn > 0):
            precision = recall = f1 = 0.
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entity-level eval. parser")

    # the pickle should be the output of build_experiment_data.py
    parser.add_argument("--data_path", help="Path to the pickle file.", type=str, required=True)

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()

    with open(args.data_path, "rb") as f:
        dataset = pickle.load(f)

    evaluator = EntityLevelEvaluator()
    evaluator.evaluate(dataset)
