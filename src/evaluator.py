from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from bert_score import BERTScorer
from rouge_score.rouge_scorer import RougeScorer
from sklearn.utils import resample
from torch.utils.data import DataLoader

from model import CandidateScorer


def get_bootstrap_ci(scores: np.ndarray, iter_: int = 10000) -> Tuple[float, float, float]:
    mean_scores = []
    for i in range(iter_):
        sample_scores = resample(scores, replace=True)
        mean_scores.append(np.mean(sample_scores))
    return np.mean(scores), np.percentile(mean_scores, 2.5), np.percentile(mean_scores, 97.5)


class Evaluator:

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
    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, Tuple[float, float, float]]:
        self.model.eval()

        candidates, summaries = [], []

        for batch in test_dataloader:
            inputs = self.__batch_to_device(batch)
            candidate_scores = self.model(doc_input_ids=inputs["doc_input_ids"],
                                          doc_att_mask=inputs["doc_att_mask"],
                                          candidates_input_ids=inputs["candidates_input_ids"],
                                          candidates_att_mask=inputs["candidates_att_mask"])
            cand_idx = np.argmax(candidate_scores.cpu().numpy(), axis=1)
            candidates.extend([cands[idx] for idx, cands in zip(cand_idx, batch["candidates"])])
            summaries.extend(batch["gt_summaries"])

        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
        bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

        rouge = [rouge_scorer.score(summary, cand) for summary, cand in zip(summaries, candidates)]
        r1 = np.array([r["rouge1"].fmeasure for r in rouge])
        r2 = np.array([r["rouge2"].fmeasure for r in rouge])
        rl = np.array([r["rougeLsum"].fmeasure for r in rouge])

        _, _, bert_f_scores = bert_scorer.score(candidates, summaries)
        bert_f_scores = bert_f_scores.cpu().numpy()

        res = {
            "rouge-1": get_bootstrap_ci(r1),
            "rouge-2": get_bootstrap_ci(r2),
            "rouge-L": get_bootstrap_ci(rl),
            "bertscore-scaled": get_bootstrap_ci(bert_f_scores),
        }

        return res
