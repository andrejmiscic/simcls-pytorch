"""
    This script aims to reproduce Figure 2 from the SimCLS paper, where the authors test model performance given
    different number of candidates.
"""

import argparse
import os
import sys

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import plotnine as gg
import torch
import torch.nn as nn

from rouge_score.rouge_scorer import RougeScorer
from torch.utils.data import DataLoader

# a hack to import from parent directory
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from evaluator import get_bootstrap_ci
from model import CandidateScorer
from utils.data_utils import SummarizationDataset, collate_inputs_to_batch
from utils.plot_utils import theme_538


class EvaluatorNumCandidates:

    def __init__(self, model: CandidateScorer) -> None:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def __batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        inputs = dict()
        for inp in batch:
            if type(batch[inp]) == torch.Tensor:
                inputs[inp] = batch[inp].to(self.device)

        return inputs

    @torch.no_grad()
    def evaluate_candidates_limits(self, test_dataloader: DataLoader, num_candidates: int) -> \
            List[Dict[str, Tuple[float, float, float]]]:
        self.model.eval()

        all_candidates = [[] for _ in range(num_candidates)]
        summaries = []

        for batch in test_dataloader:
            summaries.extend(batch["gt_summaries"])

            inputs = self.__batch_to_device(batch)
            candidate_scores = self.model(doc_input_ids=inputs["doc_input_ids"],
                                          doc_att_mask=inputs["doc_att_mask"],
                                          candidates_input_ids=inputs["candidates_input_ids"],
                                          candidates_att_mask=inputs["candidates_att_mask"])
            candidate_scores = candidate_scores.cpu().numpy()

            for i in range(num_candidates):
                cand_idx = np.argmax(candidate_scores[:, :(i + 1)], axis=1)
                all_candidates[i].extend([cands[idx] for idx, cands in zip(cand_idx, batch["candidates"])])

        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

        results = []
        for candidates in all_candidates:
            rouge = [rouge_scorer.score(summary, cand) for summary, cand in zip(summaries, candidates)]
            r1 = np.array([r["rouge1"].fmeasure for r in rouge])
            r2 = np.array([r["rouge2"].fmeasure for r in rouge])
            rl = np.array([r["rougeLsum"].fmeasure for r in rouge])

            results.append({
                "rouge-1": get_bootstrap_ci(r1),
                "rouge-2": get_bootstrap_ci(r2),
                "rouge-L": get_bootstrap_ci(rl),
            })

        return results


def parse_results_to_plot_data(results: List[Dict[str, Tuple[float, float, float]]], metric: str) -> pd.DataFrame:
    origin_res = results[0][metric]
    plot_df = pd.DataFrame({
        "rouge": [origin_res[0]] * len(results) + [r[metric][0] for r in results],
        "lower": [origin_res[1]] * len(results) + [r[metric][1] for r in results],
        "upper": [origin_res[2]] * len(results) + [r[metric][2] for r in results],
        "num_cands": list(range(1, len(results) + 1)) * 2,
        "Legend": ["Origin"] * len(results) + ["SimCLS"] * len(results)
    })
    return plot_df


def save_results_plot(plot_df: pd.DataFrame, out_dir: str, name: str) -> None:
    plot = gg.ggplot(plot_df, gg.aes(x="num_cands", y="rouge", colour="Legend")) + \
           gg.geom_line() + gg.geom_point(size=1) + \
           gg.geom_errorbar(gg.aes(ymin="lower", ymax="upper"), size=0.25) + \
           gg.xlab("Num. of test candidates") + gg.ylab(name) + theme_538() + \
           gg.theme(text=gg.element_text(size=12), legend_position=(0.85, 0.45)) + \
           gg.guides(colour=gg.guide_legend(title="Legend:"))

    gg.ggsave(plot=plot, filename=os.path.join(out_dir, f"num_cands_{name}.png"), dpi=600)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate performance vs num. candidates")

    parser.add_argument("--model_path", help="Path to the model folder", type=str, required=True)
    parser.add_argument("--test_path", help="Path to the test pickle file", type=str, required=True)
    parser.add_argument("--out_dir", help="Path to the directory to save resulting figures", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_candidates", type=int, default=16)

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()
    model = CandidateScorer(args.model_path)

    test_dataset = SummarizationDataset(args.test_path, reorder_to_original=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_inputs_to_batch)

    evaluator = EvaluatorNumCandidates(model)
    res = evaluator.evaluate_candidates_limits(test_dataloader, args.max_candidates)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    plot_df = parse_results_to_plot_data(res, metric="rouge-1")
    save_results_plot(plot_df, args.out_dir, "Rouge-1")

    plot_df = parse_results_to_plot_data(res, metric="rouge-2")
    save_results_plot(plot_df, args.out_dir, "Rouge-2")

    plot_df = parse_results_to_plot_data(res, metric="rouge-L")
    save_results_plot(plot_df, args.out_dir, "Rouge-L")


