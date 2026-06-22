#!/usr/bin/env python3
"""
eval_decoder.py

Usage examples:
    # LMK pooling with EOS as landmark (no [LMK] special token needed)
    python eval_decoder.py -m /path/to/checkpoint -base mistralai/Mistral-7B-v0.3 -peft -pool lmk -lmk_granularity 64 -task_type beir-15

    # Mean pooling
    python eval_decoder.py -m /path/to/checkpoint -base mistralai/Mistral-7B-v0.3 -peft -pool mean -task_type beir-15

    # Last-token pooling with prompting (E5-Mistral style)
    python eval_decoder.py -m /path/to/checkpoint -base mistralai/Mistral-7B-v0.3 -peft -pool last -add_prompt True -task_type beir-15
"""

import os
import json
import logging
import argparse
import shutil
import uuid
from typing import Dict, List, Union, Optional

import torch
import mteb

# ── MTEB version-safe PromptType import ───────────────────────────────────────
try:
    from mteb.encoder_interface import PromptType
except ModuleNotFoundError:
    try:
        from mteb import PromptType
    except ImportError:
        try:
            from mteb.types import PromptType
        except ImportError:
            from enum import Enum
            class PromptType(str, Enum):
                query    = "query"
                document = "passage"
            print("Could not import PromptType from mteb — using local stub.")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, models
from typing_extensions import override

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── Argument parsing ───────────────────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser()
parser.add_argument("-m",           "--model",            type=str,      required=True)
parser.add_argument("-base",        "--base_model",       type=str,      default=None)
parser.add_argument("-pool",        "--pool",             type=str,      default="last",
                    choices=["last", "mean", "lmk"])
parser.add_argument("-peft",        "--peft",             type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("-msl",         "--max_seq_len",      type=int,      default=512)
parser.add_argument("-bs",          "--batch_size",       type=int,      default=16)
parser.add_argument("-lmk_gran",    "--lmk_granularity",  type=int,      default=64)
parser.add_argument("-task_type",   "--task_type",        type=str,      default="dummy")
parser.add_argument("-lang",        "--lang",             type=str,      default="eng")
parser.add_argument("-normalize",   "--normalize",        type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("-output_dir",  "--output_dir",       type=str,      default="decoder_eval_results")
parser.add_argument("-add_prompt", "--add_prompt", type=str2bool, nargs="?", const=True, default=True,
                    help="Prepend task-specific instruct prompts to queries. "
                         "Default True — matches training which always adds instructions.")
# ── REMOVED: -add_eos is now derived automatically from pool type ──────────────
# For lmk: add_eos=False (EOS tokens are already inserted as landmarks)
# For last/mean: add_eos=True (append EOS before pooling, matches training)
args, _ = parser.parse_known_args()


# ── Task instruction map ───────────────────────────────────────────────────────

RETRIEVAL_INSTRUCTIONS: Dict[str, str] = {
    'ArguAna':                      'Given a claim, find documents that refute the claim',
    'ClimateFEVER':                 'Given a claim, retrieve documents that support or refute the claim',
    'ClimateFEVERHardNegatives':    'Given a claim, retrieve documents that support or refute the claim',
    'DBPedia':                      'Given a query, retrieve relevant entity descriptions',
    'FEVER':                        'Given a claim, retrieve documents that support or refute the claim',
    'FEVERHardNegatives':           'Given a claim, retrieve documents that support or refute the claim',
    'FiQA2018':                     'Given a financial question, retrieve user replies that best answer the question',
    'HotpotQA':                     'Given a multi-hop question, retrieve documents that can help answer the question',
    'HotpotQAHardNegatives':        'Given a multi-hop question, retrieve documents that can help answer the question',
    'MSMARCO':                      'Given a web search query, retrieve relevant passages that answer the query',
    'NFCorpus':                     'Given a question, retrieve relevant documents that best answer the question',
    'NQ':                           'Given a question, retrieve Wikipedia passages that answer the question',
    'QuoraRetrieval':               'Given a question, retrieve questions that are semantically equivalent to the given question',
    'SCIDOCS':                      'Given a statement, retrieve related passages',
    'SciFact':                      'Given a claim, retrieve documents that support or refute the claim',
    'Touche2020':                   'Given a question, retrieve passages that answer the question',
    'Touche2020Retrieval.v3':       'Given a question, retrieve passages that answer the question',
    'TRECCOVID':                    'Given a query on COVID-19, retrieve documents that answer the query',
    'LEMBNeedleRetrieval':          'Given a question, retrieve relevant passages that answer the question',
    'LEMBPasskeyRetrieval':         'Given a question, retrieve relevant passages that answer the question',
    'LEMBQMSumRetrieval':           'Given a news summary, retrieve other semantically similar summaries',
    'LEMBSummScreenFDRetrieval':    'Given a summary, retrieve other semantically similar summaries',
    'LEMBWikimQARetrieval':         'Given a question, retrieve Wikipedia passages that answer the question',
    'LEMBNarrativeQARetrieval':     'Given a question, retrieve relevant passages that answer the question',
    'MultiLongDocRetrieval':        'Given a question, retrieve relevant documents that answer the question',
    'MIRACLRetrievalHardNegatives': 'Given a question, retrieve Wikipedia passages that answer the question',
}
RETRIEVAL_INSTRUCTIONS.update({k.lower(): v for k, v in RETRIEVAL_INSTRUCTIONS.items()})
RETRIEVAL_INSTRUCTIONS.update({
    'cqadupstack':      'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question',
    'trec-covid':       RETRIEVAL_INSTRUCTIONS['TRECCOVID'],
    'climate-fever':    RETRIEVAL_INSTRUCTIONS['ClimateFEVER'],
    'dbpedia-entity':   RETRIEVAL_INSTRUCTIONS['DBPedia'],
    'webis-touche2020': RETRIEVAL_INSTRUCTIONS['Touche2020'],
    'fiqa':             RETRIEVAL_INSTRUCTIONS['FiQA2018'],
    'quora':            RETRIEVAL_INSTRUCTIONS['QuoraRetrieval'],
})
DEFAULT_INSTRUCTION = 'Given a query or question, retrieve documents that answer the query or question'


def get_instruction(task_name: str) -> str:
    # ── BUG FIX: CQADupstack check must come BEFORE the general lookup ─────────
    if task_name.lower().startswith('cqadupstack'):
        return RETRIEVAL_INSTRUCTIONS['cqadupstack']
    return RETRIEVAL_INSTRUCTIONS.get(task_name,
           RETRIEVAL_INSTRUCTIONS.get(task_name.lower(), DEFAULT_INSTRUCTION))




# ── Model merge / load ─────────────────────────────────────────────────────────

def get_merged_model(base_model_path: str, peft_model_path: str, temp_dir: str) -> str:
    """
    Merge a LoRA adapter into its base model and save to temp_dir.
    No special token handling needed — EOS is the landmark token.
    """
    logger.info(f"Loading base tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    logger.info(f"Loading base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # ── NO resize_token_embeddings needed — vocab unchanged ───────────────────

    logger.info(f"Merging adapter from {peft_model_path}")
    model = PeftModel.from_pretrained(base, peft_model_path)
    model = model.merge_and_unload()

    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    return temp_dir


def prepare_model_path() -> str:
    if args.peft:
        assert args.base_model, "--base_model is required when --peft is set"
        temp_dir = os.path.join("/tmp/temp_decoder", f"run_{str(uuid.uuid4())[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        return get_merged_model(args.base_model, args.model, temp_dir)
    logger.info(f"Loading dense checkpoint from {args.model}")
    return args.model


# ── LMK token insertion ────────────────────────────────────────────────────────

def insert_lmk_tokens(
    input_ids: List[int],
    granularity: int,
    lmk_token_id: int,   # now always eos_token_id
    max_length: int,
) -> List[int]:
    """Insert lmk_token_id at the start of every `granularity`-sized chunk."""
    result: List[int] = []
    for i, tok in enumerate(input_ids):
        if i % granularity == 0:
            result.append(lmk_token_id)
            if len(result) >= max_length:
                break
        result.append(tok)
        if len(result) >= max_length:
            break
    return result[:max_length]


# ── Sentence-Transformer subclass ─────────────────────────────────────────────

class LandmarkDecoderSentenceTransformer(SentenceTransformer):

    def __init__(self, pool: str, max_seq_len: int, lmk_granularity: int,
                 normalize: bool, add_prompt: bool, **kwargs):
        super().__init__(**kwargs)

        self.pool            = pool
        self.max_seq_len     = max_seq_len
        self.lmk_granularity = lmk_granularity
        self.normalize_emb   = normalize
        self.add_prompt      = add_prompt

        # ── BUG FIX: add_eos is derived from pool type, not a user arg ────────
        # lmk: EOS tokens are already inserted as landmarks — don't append again
        # last/mean: append EOS before pooling to match training (always_add_eos)
        self.add_eos = (pool != "lmk")

        tokenizer = self._first_module().tokenizer
        tokenizer.padding_side    = "left" if pool == "last" else "right"
        tokenizer.truncation_side = "right"

        # ── CHANGED: use unk as pad, NOT eos ─────────────────────────────────────────
        # EOS is the LMK landmark AND the sequence terminator — using it as pad causes
        # attention_mask = (input_ids != pad_id) to mask out all EOS positions, which
        # silently breaks LMK pooling (no landmarks found) and last-token pooling
        # (EOS at sequence end masked out). Training uses unk_token as pad; match that.
        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token    = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
                logger.info(f"Set pad_token = unk_token (id={tokenizer.unk_token_id}) — "
                            f"matches training; EOS remains unmasked for pooling")
            else:
                # No unk token (e.g. some Llama-3 variants) — add an explicit pad token
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                logger.warning("No unk_token found; added <pad> special token. "
                            "Ensure this matches your training tokenizer setup.")

        self.lmk_token_id: int = tokenizer.eos_token_id
        logger.info(f"Pooling={pool} | add_prompt={add_prompt} | add_eos={self.add_eos} "
                    f"| lmk_token=EOS(id={self.lmk_token_id}) | max_seq_len={max_seq_len}"
                    + (f" | lmk_granularity={lmk_granularity}" if pool == "lmk" else ""))
    # ── Pooling ────────────────────────────────────────────────────────────────

    def _last_token_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask[:, -1].all():
            return hidden[:, -1]
        seq_lens = mask.sum(dim=1) - 1
        B   = hidden.size(0)
        idx = seq_lens.clamp(0).view(B, 1, 1).expand(B, 1, hidden.size(2))
        return hidden.gather(1, idx).squeeze(1)

    def _mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp_min(1e-9)

    def _lmk_pool(self, hidden: torch.Tensor, input_ids: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        lmk_mask   = (input_ids == self.lmk_token_id) & mask.bool()
        lmk_mask_f = lmk_mask.to(hidden.dtype).unsqueeze(-1)
        lmk_sum    = (hidden * lmk_mask_f).sum(1)
        lmk_count  = lmk_mask.sum(1).clamp_min(1).unsqueeze(-1).to(hidden.dtype)
        lmk_mean   = lmk_sum / lmk_count
        no_lmk = lmk_mask.sum(1) == 0
        if no_lmk.any():
            lmk_mean[no_lmk] = self._mean_pool(hidden, mask)[no_lmk]
        return lmk_mean

    # ── Forward ───────────────────────────────────────────────────────────────

    @override
    def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        output = self._first_module().auto_model(
            **{k: features[k] for k in ["input_ids", "attention_mask"]},
            return_dict=True,
        )
        hidden    = output.last_hidden_state
        input_ids = features["input_ids"]
        mask      = features["attention_mask"]

        if self.pool == "last":
            emb = self._last_token_pool(hidden, mask)
        elif self.pool == "mean":
            emb = self._mean_pool(hidden, mask)
        elif self.pool == "lmk":
            emb = self._lmk_pool(hidden, input_ids, mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")

        if self.normalize_emb:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        features["sentence_embedding"] = emb
        return features

    # ── Tokenize ──────────────────────────────────────────────────────────────

    def _pad(self, token_lists: List[List[int]]) -> Dict[str, torch.Tensor]:
        """Pad a list of token id lists respecting tokenizer.padding_side."""
        tokenizer = self._first_module().tokenizer
        pad_id    = tokenizer.pad_token_id
        max_len   = max(len(ids) for ids in token_lists)
        if tokenizer.padding_side == "left":
            padded = [[pad_id] * (max_len - len(ids)) + ids for ids in token_lists]
        else:
            padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_lists]
        input_ids      = torch.tensor(padded, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @override
    def tokenize(self, texts: Union[List[str], List[Dict], List[tuple]],
                 **kwargs) -> Dict[str, torch.Tensor]:
        if texts and not isinstance(texts[0], str):
            texts = [str(t) for t in texts]

        tokenizer = self._first_module().tokenizer
        eos_id    = tokenizer.eos_token_id

        if self.pool == "lmk":
            # Pre-truncate to leave room for EOS landmark insertions
            budget = self.max_seq_len - (self.max_seq_len // self.lmk_granularity) - 1
            raw = tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=budget,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            # Insert EOS as landmark tokens (add_eos=False for lmk — no extra EOS appended)
            new_ids = [
                insert_lmk_tokens(ids, self.lmk_granularity, self.lmk_token_id, self.max_seq_len)
                for ids in raw["input_ids"]
            ]
            return self._pad(new_ids)

        else:
            # mean / last: optionally append EOS before padding
            eos_budget = self.max_seq_len - (1 if self.add_eos else 0)
            raw = tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=eos_budget,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            if self.add_eos:
                new_ids = [ids + [eos_id] for ids in raw["input_ids"]]
            else:
                new_ids = raw["input_ids"]
            return self._pad(new_ids)

    # ── MTEB encode interface ──────────────────────────────────────────────────

    def encode(
        self,
        sentences: List[str],
        task_name: str = "unknown",
        prompt_type: Optional[PromptType] = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        **kwargs,
    ):
        is_query = (prompt_type is None or prompt_type != PromptType.document)
        if self.add_prompt and is_query:
            sentences = [self.prompts['query']+s for s in sentences]
            logger.info(f"[{task_name}] Prompt applied. Example: {sentences[0][:120]!r}")

        return super().encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )


# ── MTEB task map ──────────────────────────────────────────────────────────────

MULTILINGUAL_TASK_NAMES = [
    "MultiLongDocRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MultiEURLEXMultilabelClassification",
]

TASK_MAP = {
    "dummy":              ["SciFact"],
    "mteb_eng_benchmark": ["MTEB(eng, v2)"],
    "mteb_v2":            ["ArguAna","FiQA2018","SCIDOCS","CQADupstackUnixRetrieval",
                           "CQADupstackGamingRetrieval","ClimateFEVERHardNegatives",
                           "FEVERHardNegatives","HotpotQAHardNegatives","TRECCOVID",
                           "Touche2020Retrieval.v3"],
    "beir-15":            ["NFCorpus","NQ","HotpotQA","Touche2020","CQADupstackRetrieval",
                           "QuoraRetrieval","DBPedia","FEVER","ClimateFEVER","SciFact"],
    "msmarco":            ["MSMARCO"],
    "mldr":               ["MultiLongDocRetrieval"],
    "miracl_hn":          ["MIRACLRetrievalHardNegatives"],
    "long_embed":         ["LEMBNeedleRetrieval","LEMBPasskeyRetrieval","LEMBQMSumRetrieval",
                           "LEMBSummScreenFDRetrieval","LEMBWikimQARetrieval","LEMBNarrativeQARetrieval"],
    "coir":               ["AppsRetrieval","CodeFeedbackMT","CodeFeedbackST",
                           "CodeTransOceanContest","CodeTransOceanDL","CosQA",
                           "SyntheticText2SQL","StackOverflowQA",
                           "COIRCodeSearchNetRetrieval","CodeSearchNetCCRetrieval"],
}


def get_task_names(task_type: str) -> List[str]:
    return TASK_MAP.get(task_type, [task_type])


def get_main_score(out) -> float:
    scores = out[0].scores
    for split in ("test", "dev", "train"):
        if split in scores:
            return scores[split][0]["main_score"]
    return float("nan")


def build_run_tag() -> str:
    model_tag  = "_".join(args.model.rstrip("/").split("/")[-3:])
    pool_tag   = args.pool + (f"_gran{args.lmk_granularity}" if args.pool == "lmk" else "")
    prompt_tag = "prompt" if args.add_prompt else "noprompt"
    return f"{args.lang}_{pool_tag}_{prompt_tag}_{args.max_seq_len}_{model_tag}"


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    run_tag    = build_run_tag()
    task_names = get_task_names(args.task_type)
    logger.info(f"Run tag : {run_tag}")
    logger.info(f"Tasks   : {task_names}")

    model_path = prepare_model_path()
    is_temp    = args.peft

    word_embedding_model = models.Transformer(
        model_path,
        max_seq_length=args.max_seq_len,
        config_args={"trust_remote_code": True},
        model_args={"trust_remote_code": True, "torch_dtype": torch.bfloat16},
    )
    model = LandmarkDecoderSentenceTransformer(
        modules         = [word_embedding_model],
        pool            = args.pool,
        max_seq_len     = args.max_seq_len,
        lmk_granularity = args.lmk_granularity,
        normalize       = args.normalize,
        add_prompt      = args.add_prompt,
    )

    if is_temp:
        logger.info(f"Cleaning up {model_path}")
        shutil.rmtree(model_path)

    if task_names == ["MTEB(eng, v2)"]:
        tasks      = mteb.get_benchmark("MTEB(eng, v2)")
        output_dir = os.path.join(args.output_dir, f"mtebengbenchmark_{run_tag}")
        os.makedirs(output_dir, exist_ok=True)
        out = mteb.evaluate(model=model, tasks=tasks, overwrite_strategy="always",
                            encode_kwargs={"batch_size": args.batch_size})
        for tr in out.task_results:
            if hasattr(tr, "scores") and tr.scores is not None:
                with open(os.path.join(output_dir, f"{tr.task_name}.json"), "w") as f:
                    json.dump(tr.scores, f, indent=2)
        logger.info("MTEB benchmark complete.")
        return

    for task_name in task_names:
        eval_splits = ["dev"] if any(k in task_name.lower() for k in ("msmarco", "miracl")) else ["test"]
        eval_splits = eval_splits if "lemb" not in task_name.lower() else None
        languages   = [args.lang] if task_name in MULTILINGUAL_TASK_NAMES else None

        task = (mteb.get_task(task_name, languages=languages, eval_splits=eval_splits)
                if languages else mteb.get_task(task_name))

        output_dir = os.path.join(args.output_dir, f"{task_name}_{run_tag}")
        os.makedirs(output_dir, exist_ok=True)


        logger.info(f"Prompts before {task_name}: {model.prompts}")
        if args.add_prompt:
            new_prompts={"query": f"Instruct: {get_instruction(task_name)}\nQuery: ", "document": ""}
            model.prompts.update(new_prompts)
        logger.info(f"Prompts after {task_name}: {model.prompts}")

        out = mteb.evaluate(
            model=model,
            tasks=[task] if not isinstance(task, list) else task,
            overwrite_strategy="always",
            encode_kwargs={"batch_size": 1 if "lemb" in task_name.lower() else args.batch_size},
        )

        result_file = os.path.join(output_dir, f"{task_name}.json")
        if len(out.task_results) > 1:
            with open(result_file, "w") as f:
                json.dump([tr.scores for tr in out.task_results
                           if hasattr(tr, "scores") and tr.scores is not None], f, indent=2)
        else:
            with open(result_file, "w") as f:
                json.dump(out.task_results[0].scores, f, indent=2)

        logger.info(f"{task_name:40s}  main_score = {get_main_score(out):.4f}")


if __name__ == "__main__":
    main()