from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rouge_score.rouge_scorer import RougeScorer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CandidateScorer


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    num_epochs: int
    save_dir: str = None
    weight_decay: float = 0
    margin_lambda: float = 0.01
    warmup_steps: int = 10000  # as described in the paper
    eval_steps: int = 1000  # in the original work they evaluate every 1000 updates
    early_stopping_patience: int = -1  # -1 don't use early stopping


class RankingLoss(nn.Module):

    def __init__(self, margin_lambda: float = 0.01) -> None:
        super(RankingLoss, self).__init__()

        self.margin_lambda = margin_lambda

    def forward(self, candidates_scores: torch.Tensor, summary_scores: torch.Tensor) -> torch.Tensor:

        batch_size, num_candidates = candidates_scores.size()

        # computes candidates vs summary loss
        summary_scores = summary_scores.unsqueeze(1).expand(batch_size, num_candidates)
        ranking_target = torch.ones_like(candidates_scores)
        loss = F.margin_ranking_loss(summary_scores, candidates_scores, target=ranking_target, margin=0.)

        # computes candidates ranking loss
        for i in range(1, num_candidates):
            ranking_target = torch.ones_like(candidates_scores[:, :-i])
            loss += F.margin_ranking_loss(candidates_scores[:, :-i], candidates_scores[:, i:],
                                          target=ranking_target, margin=i * self.margin_lambda)

        return loss


class Scheduler:
    """
        SimCLS learning rate scheduler as described in paper.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_lambda: float = 0.002, warmup_steps: int = 10000) -> None:
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.warmup_steps = warmup_steps

        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr_lambda * min(pow(self.step_count, -0.5), self.step_count * pow(self.warmup_steps, -1.5))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class Trainer:
    def __init__(self, model: CandidateScorer) -> None:

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.best_score = -float("inf")
        self.worse_count = 0

    def __batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        inputs = dict()
        for inp in batch:
            if type(batch[inp]) == torch.Tensor:
                inputs[inp] = batch[inp].to(self.device)

        return inputs

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, config: TrainConfig) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = Scheduler(optimizer, config.lr, config.warmup_steps)
        criterion = RankingLoss(config.margin_lambda)

        step_counter = 0
        eval_steps_loss_sum = 0
        self.best_score = -float("inf")
        self.worse_count = 0

        for epoch in range(config.num_epochs):
            epoch_loss_sum = 0

            for i, batch in enumerate(tqdm(train_dataloader)):
                step_counter += 1

                self.model.train()
                inputs = self.__batch_to_device(batch)  # send to GPU

                optimizer.zero_grad()

                candidate_scores, summary_scores = self.model(**inputs)
                loss = criterion(candidate_scores, summary_scores)

                loss.backward()
                scheduler.step()
                optimizer.step()

                epoch_loss_sum += loss.item()
                eval_steps_loss_sum += loss.item()

                if step_counter % config.eval_steps == 0:
                    r1, r2, rl, val_loss = self.evaluate(val_dataloader, criterion)
                    val_score = (r1 + r2 + rl) / 3
                    train_loss = eval_steps_loss_sum / config.eval_steps
                    eval_steps_loss_sum = 0

                    print(f"[INFO] After {step_counter} steps:\n\t- train loss: {train_loss:.6f}"
                          f"\n\t- val loss: {val_loss:.4f}, rouge scores: {r1:.4f}/{r2:.4f}/{rl:.4f}")

                    should_early_stop = self.__handle_early_stopping(val_score, config.early_stopping_patience,
                                                                     config.save_dir)
                    if should_early_stop:
                        return

            print(f"[INFO] Average loss in epoch {epoch}: {epoch_loss_sum / len(train_dataloader)}")

        # after training is finished, we re-evaluate the model
        r1, r2, rl, val_loss = self.evaluate(val_dataloader, criterion)
        val_score = (r1 + r2 + rl) / 3
        print(f"[INFO] After {step_counter} steps:"
              f"\n\t- val loss: {val_loss:.4f}, rouge scores: {r1:.4f}/{r2:.4f}/{rl:.4f}")
        self.__handle_early_stopping(val_score, config.early_stopping_patience, config.save_dir)

    def evaluate(self, test_dataloader: DataLoader, criterion: RankingLoss) -> Tuple[float, float, float, float]:
        self.model.eval()
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

        losses = []
        r1, r2, rl = [], [], []
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = self.__batch_to_device(batch)

                candidate_scores, summary_scores = self.model(**inputs)
                loss = criterion(candidate_scores, summary_scores)
                losses.append(loss.item())

                cand_idx = np.argmax(candidate_scores.cpu().numpy(), axis=1)
                candidates = [cands[idx] for idx, cands in zip(cand_idx, batch["candidates"])]
                res = [rouge_scorer.score(summary, cand) for summary, cand in zip(batch["gt_summaries"], candidates)]

                r1.extend([r["rouge1"].fmeasure for r in res])
                r2.extend([r["rouge2"].fmeasure for r in res])
                rl.extend([r["rougeLsum"].fmeasure for r in res])

        return np.mean(r1), np.mean(r2), np.mean(rl), np.mean(losses)

    def __handle_early_stopping(self, val_score: float, early_stopping_patience: int, save_dir: str) -> bool:
        if val_score > self.best_score or early_stopping_patience == -1:
            self.best_score = val_score
            self.worse_count = 0

            if hasattr(self.model, "module"):
                self.model.module.save(save_dir)
            else:
                self.model.save(save_dir)
            return False
        else:
            self.worse_count += 1
            return self.worse_count >= early_stopping_patience
