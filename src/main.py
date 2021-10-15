import argparse
import os

import torch

from torch.utils.data import DataLoader

from evaluator import Evaluator
from model import CandidateScorer
from trainer import Trainer, TrainConfig
from utils.data_utils import SummarizationDataset, collate_inputs_to_batch


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimCLS training/evaluation parser")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    parser.add_argument("--model_path", help="Path to the model folder (can be huggingface hub, e.g. roberta-base)",
                        type=str, default="roberta-base")
    parser.add_argument("--train_path", help="Path to the training pickle file", type=str, default=None)
    parser.add_argument("--val_path", help="Path to the val pickle file", type=str, default=None)
    parser.add_argument("--test_path", help="Path to the test pickle file", type=str, default=None)
    parser.add_argument("--save_dir", help="Path to the directory to output the trained model", type=str, default=None)

    # training parameters
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)  # also used for testing
    parser.add_argument("--margin_lambda", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=-1, help="-1 to not perform early stopping")

    return parser


def parse_args_to_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, save_dir=args.save_dir,
                       weight_decay=args.weight_decay, margin_lambda=args.margin_lambda, eval_steps=args.eval_steps,
                       early_stopping_patience=args.early_stop_patience)


if __name__ == '__main__':
    torch.manual_seed(0)
    args = setup_parser().parse_args()

    model = CandidateScorer(args.model_path)

    if args.mode == "train":
        assert args.train_path, "If you want to train the model, you need to provide --train_path argument."
        assert args.val_path, "If you want to train the model, you need to provide --val_path argument."
        assert args.save_dir, "If you want to train the model, you need to provide --save_dir argument."

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        config = parse_args_to_config(args)

        train_dataset = SummarizationDataset(args.train_path)
        val_dataset = SummarizationDataset(args.val_path)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_inputs_to_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=2*config.batch_size, shuffle=False, collate_fn=collate_inputs_to_batch)

        trainer = Trainer(model)
        trainer.train(train_dataloader, val_dataloader, config)

    elif args.mode == "test":
        assert args.test_path, "If you want to test the model, you need to provide --test_path argument."

        test_dataset = SummarizationDataset(args.test_path)
        test_dataloader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_inputs_to_batch)

        evaluator = Evaluator(model)
        res = evaluator.evaluate(test_dataloader)

        print("SIMCLS RESULTS:")
        for metric, results in res.items():
            print(f"\t- {metric}: {results[0]}, [{results[1]}, {results[2]}]")


