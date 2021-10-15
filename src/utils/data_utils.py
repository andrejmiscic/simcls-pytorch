import pickle

from typing import Dict, List, Any

import numpy as np
import torch


class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, reorder_to_original=False) -> None:
        """
        :param reorder_to_original: if true, the candidates will be reordered as they were originally generated, not
                                    according to their Rouge scores, used to select main candidate
        """
        with open(path, "rb") as f:
            self.inputs = pickle.load(f)

        if reorder_to_original:
            self.__reorder_inputs()

    def __reorder_inputs(self) -> None:
        reordered_inputs = []
        for datapoint in self.inputs:
            original_order = np.argsort(datapoint["candidate_order"])
            datapoint["candidates"] = [datapoint["candidates"][o] for o in original_order]
            datapoint["candidates_input_ids"] = [datapoint["candidates_input_ids"][o] for o in original_order]
            datapoint["candidates_att_mask"] = [datapoint["candidates_att_mask"][o] for o in original_order]
            reordered_inputs.append(datapoint)

        self.inputs = reordered_inputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.inputs[idx]


def collate_inputs_to_batch(inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {
        "docs": [inp["doc"] for inp in inputs],
        "gt_summaries": [inp["gt_summary"] for inp in inputs],
        "candidates": [inp["candidates"] for inp in inputs],
        "doc_input_ids": torch.tensor([inp["doc_input_ids"] for inp in inputs], dtype=torch.long),
        "doc_att_mask": torch.tensor([inp["doc_att_mask"] for inp in inputs], dtype=torch.long),
        "candidates_input_ids": torch.tensor([inp["candidates_input_ids"] for inp in inputs], dtype=torch.long),
        "candidates_att_mask": torch.tensor([inp["candidates_att_mask"] for inp in inputs], dtype=torch.long),
        "summary_input_ids": torch.tensor([inp["summary_input_ids"] for inp in inputs], dtype=torch.long),
        "summary_att_mask": torch.tensor([inp["summary_att_mask"] for inp in inputs], dtype=torch.long)
    }

    return batch
