import argparse
import os
import pickle

from typing import Tuple, List, Dict, Any

import numpy as np
import torch

from datasets import load_dataset
from nltk import sent_tokenize
from rouge_score.rouge_scorer import RougeScorer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaTokenizer

from model import CandidateGenerator


def __fix_cnndm(text: str) -> str:
    # for some reason cnndm texts on huggingface datasets hub sometimes contain spaces before periods
    return text.replace(" .", ".")


def get_data(dataset: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    if dataset == "cnndm":
        cnndm = load_dataset("cnn_dailymail", "3.0.0")
        train = [{"doc": __fix_cnndm(d["article"]), "summary": __fix_cnndm(d["highlights"])} for d in cnndm["train"]]
        val = [{"doc": __fix_cnndm(d["article"]), "summary": __fix_cnndm(d["highlights"])} for d in cnndm["validation"]]
        test = [{"doc": __fix_cnndm(d["article"]), "summary": __fix_cnndm(d["highlights"])} for d in cnndm["test"]]
    elif dataset == "xsum":
        xsum = load_dataset("xsum")
        train = [{"doc": d["document"], "summary": d["summary"]} for d in xsum["train"]]
        val = [{"doc": d["document"], "summary": d["summary"]} for d in xsum["validation"]]
        test = [{"doc": d["document"], "summary": d["summary"]} for d in xsum["test"]]
    elif dataset == "billsum":
        billsum = load_dataset("billsum")
        train_val = [{"doc": d["text"], "summary": d["summary"]} for d in billsum["train"]]
        train, val = train_test_split(train_val, test_size=0.1, random_state=0)
        test = [{"doc": d["text"], "summary": d["summary"]} for d in billsum["test"]]
    else:
        raise NotImplementedError(f"{dataset} dataset not supported.")

    return train, val, test


def build_dataset(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:

    dataset = []

    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:(i + args.batch_size)]
        docs = [inp["doc"] for inp in batch]
        summaries = [inp["summary"] for inp in batch]

        batched_candidates = generator(docs)

        for candidates, summary, doc in zip(batched_candidates, summaries, docs):
            # a requirement by the rouge_score library is that sentences in input text are separated by a newline
            # this is due to the computation of Rouge-L ("rougeLsum")
            # lower-case the sentences as they do in the paper
            doc = "\n".join([sent.lower() for sent in sent_tokenize(doc)])
            summary = "\n".join([sent.lower() for sent in sent_tokenize(summary)])
            candidates = ["\n".join([sent.lower() for sent in sent_tokenize(cand)]) for cand in candidates]

            scores = []
            for cand in candidates:
                rouge = rouge_scorer.score(summary, cand)
                # each candidate is assigned a score - mean of Rouge-1, Rouge-2 and Rouge-L
                scores.append((rouge["rouge1"].fmeasure + rouge["rouge2"].fmeasure + rouge["rougeLsum"].fmeasure) / 3)

            order = list(reversed(np.argsort(scores).tolist()))
            ordered_candidates = [candidates[o] for o in order]

            doc_inputs = tokenizer(doc, max_length=512, padding="max_length", truncation=True)
            summary_inputs = tokenizer(summary, max_length=args.cand_max_len, padding="max_length", truncation=True)
            candidates_inputs = tokenizer(ordered_candidates, max_length=args.cand_max_len,
                                          padding="max_length", truncation=True)

            new_datapoint = {
                "doc": doc, "gt_summary": summary, "candidates": ordered_candidates,
                "candidate_order": order,
                "doc_input_ids": doc_inputs["input_ids"],
                "doc_att_mask": doc_inputs["attention_mask"],
                "candidates_input_ids": candidates_inputs["input_ids"],
                "candidates_att_mask": candidates_inputs["attention_mask"],
                "summary_input_ids": summary_inputs["input_ids"],
                "summary_att_mask": summary_inputs["attention_mask"]
            }
            dataset.append(new_datapoint)

    return dataset


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset builder argument parser")
    parser.add_argument("--dataset", type=str, required=True, choices=["cnndm", "xsum", "billsum"])
    parser.add_argument("--gen_type", type=str, required=True, choices=["Bart", "Pegasus"])
    parser.add_argument("--gen_path", type=str, required=True,
                        help="Path to the gen. model folder (can be HF hub, e.g. facebook/bart-large-xsum)")
    parser.add_argument("--out_dir", help="Path to the output directory", type=str, required=True)
    parser.add_argument("--tok_path", help="Path to the tokenizer folder (HF hub)", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cand_max_len", type=int, default=120,
                        help="Max length of candidate summary, from SimCLS repo: 120 for CNN-DM, 80 for X-SUM")

    # parameters for candidate generation
    # you can read more about these parameters here:
    # https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
    parser.add_argument("--num_cands", type=int, default=16)
    parser.add_argument("--num_beam_groups", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=16)
    parser.add_argument("--no_repeat_ngram", type=int, default=3)
    parser.add_argument("--diversity_penalty", type=float, default=1.)
    parser.add_argument("--length_penalty", type=float, default=2.)

    return parser


if __name__ == '__main__':
    torch.manual_seed(0)
    args = setup_parser().parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    generator = CandidateGenerator(generator_type=args.gen_type, path=args.gen_path, num_return_seqs=args.num_cands,
                                   num_beam_groups=args.num_beam_groups, num_beams=args.num_beams,
                                   no_repeat_ngram_n=args.no_repeat_ngram, diversity_penalty=args.diversity_penalty,
                                   length_penalty=args.length_penalty)
    tokenizer = RobertaTokenizer.from_pretrained(args.tok_path)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    train, val, test = get_data(args.dataset)

    test_dataset = build_dataset(test)
    with open(os.path.join(args.out_dir, f"{args.dataset}_test.pkl"), "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"[INFO] Saved test dataset to {args.out_dir}")

    val_dataset = build_dataset(val)
    with open(os.path.join(args.out_dir, f"{args.dataset}_val.pkl"), "wb") as f:
        pickle.dump(val_dataset, f)
    print(f"[INFO] Saved val dataset to {args.out_dir}")

    train_dataset = build_dataset(train)
    with open(os.path.join(args.out_dir, f"{args.dataset}_train.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)

    print(f"[INFO] Saved train dataset to {args.out_dir}")

