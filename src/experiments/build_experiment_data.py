"""
    This script is used to build the [doc, summary, origin candidate, predicted candidate] tuple for each datapoint in
    the test set. The built data can then be used for entity-level and sentence-level evaluation as presented in
    the SimCLS paper.
"""

import argparse
import os
import pickle
import sys

from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# a hack to import from parent directory
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.data_utils import SummarizationDataset, collate_inputs_to_batch
from model import CandidateScorer


class ExperimentDataBuilder:

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
    def build(self, test_dataloader: DataLoader) -> List[Dict[str, str]]:
        self.model.eval()

        data = []

        for batch in test_dataloader:
            inputs = self.__batch_to_device(batch)
            candidate_scores = self.model(doc_input_ids=inputs["doc_input_ids"],
                                          doc_att_mask=inputs["doc_att_mask"],
                                          candidates_input_ids=inputs["candidates_input_ids"],
                                          candidates_att_mask=inputs["candidates_att_mask"])
            cand_idx = np.argmax(candidate_scores.cpu().numpy(), axis=1)
            candidates = [cands[idx] for idx, cands in zip(cand_idx, batch["candidates"])]
            origin_candidates = [cands[0] for cands in batch["candidates"]]

            data.extend([{"doc": doc, "summary": summ, "origin_candidate": orig_cand, "predicted_candidate": pred_cand}
                         for doc, summ, orig_cand, pred_cand in
                         zip(batch["docs"], batch["gt_summaries"], origin_candidates, candidates)])

        return data


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build experiment data parser")

    parser.add_argument("--model_path", help="Path to the model folder", type=str, required=True)
    parser.add_argument("--test_path", help="Path to the test pickle file", type=str, required=True)
    parser.add_argument("--out_path", help="Path to the output pickle file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=48)

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()
    model = CandidateScorer(args.model_path)

    test_dataset = SummarizationDataset(args.test_path, reorder_to_original=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_inputs_to_batch)

    experiment_data_builder = ExperimentDataBuilder(model)
    data = experiment_data_builder.build(test_dataloader)

    with open(args.out_path, "wb") as f:
        pickle.dump(data, f)
