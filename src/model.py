from enum import Enum
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaTokenizer, BartTokenizer, BartForConditionalGeneration, \
    PegasusTokenizer, PegasusForConditionalGeneration, PreTrainedModel, PreTrainedTokenizer


class GeneratorType(Enum):
    Bart = 0
    Pegasus = 1


class CandidateGenerator:
    """
        CandidateGenerator is not trainable, it's just a wrapper around a pretrained model for summarization.
    """

    def __init__(self, generator_type: Union[str, GeneratorType], path: str, num_return_seqs: int = 16,
                 num_beam_groups: int = 16, num_beams: int = 16, no_repeat_ngram_n: int = 3,
                 diversity_penalty: float = 1., length_penalty: float = 2., device: torch.device = None) -> None:
        super(CandidateGenerator, self).__init__()

        self.num_return_seqs, self.num_beam_groups, self.num_beams, self.no_repeat_ngram_n = \
            num_return_seqs, num_beam_groups, num_beams, no_repeat_ngram_n
        self.diversity_penalty, self.length_penalty = diversity_penalty, length_penalty
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator_type = GeneratorType[generator_type] if type(generator_type) == str else generator_type
        self.generator, self.tokenizer = self.__get_generator_and_tokenizer(path)

        self.generator = self.generator.eval().to(self.device)

    @torch.no_grad()
    def forward(self, docs: List[str], **kwargs) -> List[List[str]]:

        inputs = self.tokenizer(docs, padding="longest", truncation=True, return_tensors="pt")

        candidates_input_ids = self.generator.generate(
            input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device),
            early_stopping=True,
            num_beams=kwargs["num_beams"] if "num_beams" in kwargs else self.num_beams,
            length_penalty=kwargs["length_penalty"] if "length_penalty" in kwargs else self.length_penalty,
            no_repeat_ngram_size=kwargs["no_repeat_ngram_n"] if "no_repeat_ngram_n" in kwargs else self.no_repeat_ngram_n,
            num_return_sequences=kwargs["num_return_seqs"] if "num_return_seqs" in kwargs else self.num_return_seqs,
            num_beam_groups=kwargs["num_beam_groups"] if "num_beam_groups" in kwargs else self.num_beam_groups,
            diversity_penalty=kwargs["diversity_penalty"] if "diversity_penalty" in kwargs else self.diversity_penalty,
        )

        batched_cands = self.tokenizer.batch_decode(candidates_input_ids, skip_special_tokens=True)
        cands_per_doc = len(batched_cands) // len(docs)
        candidates = [batched_cands[(i * cands_per_doc):((i + 1) * cands_per_doc)] for i in range(len(docs))]
        return candidates

    def __call__(self, docs: List[str], **kwargs) -> List[List[str]]:
        return self.forward(docs, **kwargs)

    def __get_generator_and_tokenizer(self, path: str = None) -> \
            Tuple[PreTrainedModel, PreTrainedTokenizer]:

        if self.generator_type == GeneratorType.Bart:
            return BartForConditionalGeneration.from_pretrained(path), BartTokenizer.from_pretrained(path)
        elif self.generator_type == GeneratorType.Pegasus:
            return PegasusForConditionalGeneration.from_pretrained(path), PegasusTokenizer.from_pretrained(path)
        else:
            raise NotImplementedError(f"Generator type: {self.generator_type} not supported.")


class CandidateScorer(nn.Module):

    def __init__(self, path: str) -> None:
        super(CandidateScorer, self).__init__()

        self.encoder = RobertaModel.from_pretrained(path)

    def forward(self, doc_input_ids: torch.Tensor, doc_att_mask: torch.Tensor, candidates_input_ids: torch.Tensor,
                candidates_att_mask: torch.Tensor, summary_input_ids: torch.Tensor = None,
                summary_att_mask: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # calculates document representation
        doc_embs = self.encoder(input_ids=doc_input_ids, attention_mask=doc_att_mask)[0]
        doc_cls = doc_embs[:, 0, :]  # CLS is the first token in the sequence

        # calculates representations of all candidate summaries
        # encoder expects a two-dimensional input tensor
        batch_size, num_candidates, seq_len = candidates_input_ids.size()

        candidates_input_ids = candidates_input_ids.reshape(batch_size * num_candidates, seq_len)
        candidates_att_mask = candidates_att_mask.reshape(batch_size * num_candidates, seq_len)
        candidates_embs = self.encoder(input_ids=candidates_input_ids, attention_mask=candidates_att_mask)[0]
        candidates_cls = candidates_embs[:, 0, :].reshape(batch_size, num_candidates, -1)

        if summary_input_ids is None:
            doc_cls = doc_cls.reshape(batch_size, 1, -1).expand(batch_size, num_candidates, -1)
            return torch.cosine_similarity(doc_cls, candidates_cls, dim=-1)

        # calculates reference summary representation
        summary_embs = self.encoder(input_ids=summary_input_ids, attention_mask=summary_att_mask)[0]
        summary_cls = summary_embs[:, 0, :]

        ref_summary_scores = torch.cosine_similarity(doc_cls, summary_cls, dim=-1)
        doc_cls = doc_cls.reshape(batch_size, 1, -1).expand(batch_size, num_candidates, -1)
        candidates_scores = torch.cosine_similarity(doc_cls, candidates_cls, dim=-1)

        return candidates_scores, ref_summary_scores

    def save(self, path: str) -> None:
        self.encoder.save_pretrained(path)


class SimCLS:
    """
        SimCLS is not meant for training, rather it's a wrapper around summary generator and scorer for easier usage.
    """

    def __init__(self, generator_type: Union[str, GeneratorType], generator_path: str, scorer_path: str,
                 scorer_tokenizer_path: str = None, candidate_max_len: int = 120, **kwargs) -> None:

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator_type = GeneratorType[generator_type] if type(generator_type) == str else generator_type
        self.candidate_max_len = candidate_max_len

        self.candidate_generator = CandidateGenerator(generator_type, generator_path, device=self.device, **kwargs)

        self.candidate_scorer = CandidateScorer(scorer_path).eval().to(self.device)
        tok_path = scorer_tokenizer_path if scorer_tokenizer_path else scorer_path
        self.scorer_tokenizer = RobertaTokenizer.from_pretrained(tok_path)

    def forward(self, docs: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:

        docs = [docs] if type(docs) == str else docs
        candidates = self.candidate_generator(docs, **kwargs)

        # lower-case inputs as instructed in the paper (roberta-base is case-sensitive)
        docs = [doc.lower() for doc in docs]
        candidates_lower = [cand.lower() for doc_cands in candidates for cand in doc_cands]
        doc_inputs = self.scorer_tokenizer(docs, padding="longest", truncation=True, return_tensors="pt")
        candidate_inputs = self.scorer_tokenizer(candidates_lower, max_length=self.candidate_max_len,
                                                 padding="longest", truncation=True, return_tensors="pt")

        num_docs, num_candidates_per_doc = len(docs), len(candidates[0])
        candidates_input_ids = candidate_inputs["input_ids"].reshape(num_docs, num_candidates_per_doc, -1)
        candidates_attention_mask = candidate_inputs["attention_mask"].reshape(num_docs, num_candidates_per_doc, -1)

        with torch.no_grad():
            scores = self.candidate_scorer(doc_input_ids=doc_inputs["input_ids"],
                                           doc_att_mask=doc_inputs["attention_mask"],
                                           candidates_input_ids=candidates_input_ids,
                                           candidates_att_mask=candidates_attention_mask)
            scores = scores.cpu().numpy()

        indices = np.argmax(scores, axis=-1)
        summaries = [doc_cands[i] for i, doc_cands in zip(indices, candidates)]
        summaries = np.squeeze(summaries).tolist()

        return summaries

    def __call__(self, docs: Union[str, List[str]]) -> Union[str, List[str]]:
        return self.forward(docs)

    def __get_generator_tokenizer(self, generator_path) -> PreTrainedTokenizer:
        if self.generator_type == GeneratorType.Bart:
            return BartTokenizer.from_pretrained(generator_path)
        elif self.generator_type == GeneratorType.Pegasus:
            return PegasusTokenizer.from_pretrained(generator_path)
        else:
            raise NotImplementedError(f"Generator type: {self.generator_type} not supported.")

