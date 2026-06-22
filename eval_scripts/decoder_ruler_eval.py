#!/usr/bin/env python3
"""
eval_ruler_full_doc.py  —  RULER-aligned, adapted for full-document retrieval

Data construction follows RULER exactly:
  Haystack : Paul Graham essays  (PaulGrahamEssays.json)
  Keys     : English words       (english_words.json)
  Needle   : "The special magic {type_needle_v} for {key} is {value}."

NIAH subtasks (following RULER):
  niah_single_1   : 1 needle,  value = 7-digit number
  niah_single_2   : 1 needle,  value = English word
  niah_single_3   : 1 needle,  value = UUID
  niah_multikey_1 : 4  distinct (key,value) pairs in doc; query picks 1
  niah_multikey_2 : 8  distinct (key,value) pairs in doc; query picks 1
  niah_multikey_3 : 16 distinct (key,value) pairs in doc; query picks 1
  niah_multivalue : 4  values for the same key; all are gold
  niah_multiquery : 8  pairs in doc; each example queries a different one

CWE (Common Words Extraction) — RULER-aligned:
  Entire document = single numbered list: "1. apple\n2. bird\n3. apple\n..."
  Gold doc: target words each appear N_COMMON_OCCURRENCES times
  Distractor: same format, no elevated word
  Depth: position of first common word in the list
  Query: "Find the document where these words are most common: [w1, w2, w3]"

FWE (Frequent Words Extraction) — RULER-aligned:
  Entire document = coded text: "xkd....mpt....xkd....qzw....xkd...."
  Gold doc: target coded words appear FWE_FREQUENCY times
  Distractor: same format, different frequent word
  Depth: position of first occurrence of target coded word
  Query: "Find the document where these coded words appear most frequently: [xkd, mpt]"

Usage:
  python eval_ruler_full_doc.py \\
      --model /path/to/merged_checkpoint \\
      --pool lmk --lmk_granularity 64 \\
      --haystack_file PaulGrahamEssays.json \\
      --english_words_file english_words.json \\
      --lengths 4096 8192 16384 32768 65536 131072 \\
      --depths 0.0 0.25 0.5 0.75 1.0 \\
      --corpus_size 20 --num_examples 25 \\
      --output_dir ruler_results
"""

import os, re, csv, json, math, random, string, shutil, logging, argparse, uuid as _uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _bidirectional_expand_mask(mask, dtype, tgt_len=None):
    if mask.dim() == 2:
        bsz, src_len = mask.size()
        tgt_len = tgt_len or src_len
        expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    elif mask.dim() == 3:
        expanded = mask.unsqueeze(1).to(dtype)
    else:
        raise ValueError(f"Cannot expand mask shape: {mask.shape}")
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)

AttentionMaskConverter._expand_mask = staticmethod(_bidirectional_expand_mask)


from sentence_transformers import SentenceTransformer, models

# ── CLI ────────────────────────────────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','1'): return True
    if v.lower() in ('no','false','f','0'): return False
    raise argparse.ArgumentTypeError('Boolean expected.')

ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                             description=__doc__)
# Model
ap.add_argument('--model',              type=str,      required=True)
ap.add_argument('--base_model',         type=str,      default=None)
ap.add_argument('--peft',               type=str2bool, nargs='?', const=True, default=False)
ap.add_argument('--pool',               type=str,      default='lmk',
                choices=['last','mean','lmk'])
ap.add_argument('--lmk_granularity',    type=int,      default=64)
# Data
ap.add_argument('--haystack_file',      type=str,      default='PaulGrahamEssays.json',
                help='PaulGrahamEssays.json produced by download_paulgraham_essay.py')
ap.add_argument('--english_words_file', type=str,      default='english_words.json',
                help='english_words.json from RULER repo (list of English words)')
# Grid
ap.add_argument('--lengths',            type=int,      nargs='+',
                default=[512, 1024, 2048, 4096, 8192, 16384])
                # default=[4096, 8192, 16384, 32768, 65536, 131072])
ap.add_argument('--depths',             type=float,    nargs='+',
                default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], #[0.0, 0.25, 0.5, 0.75, 1.0],
                help='Needle depth as fraction of document (0=start, 1=end)')
ap.add_argument('--tasks',              type=str,      nargs='+',
                default=['niah_single_1','niah_single_2','niah_single_3',
                         'niah_multikey_1','niah_multikey_2','niah_multikey_3',
                         'niah_multivalue','niah_multiquery',
                        #  'cwe','fwe'
                         ])
# Corpus
ap.add_argument('--num_examples',       type=int,      default=25,
                help='Queries per (task, length, depth) cell')
ap.add_argument('--corpus_size',        type=int,      default=20,
                help='Documents per corpus (1 gold + corpus_size-1 distractors)')
# Task-specific
ap.add_argument('--cwe_num_common',     type=int,      default=3,
                help='Number of target common words in CWE gold document')
ap.add_argument('--cwe_occurrences',    type=int,      default=10,
                help='How many times each common word appears in CWE gold doc')
ap.add_argument('--fwe_num_targets',    type=int,      default=3,
                help='Number of target coded words for FWE')
ap.add_argument('--fwe_frequency',      type=int,      default=15,
                help='How many times each target coded word appears in FWE gold doc')
# Encoding
ap.add_argument('--batch_size',         type=int,      default=1)
ap.add_argument('--add_prompt',         type=str2bool, nargs='?', const=True, default=True)
ap.add_argument('--normalize',          type=str2bool, nargs='?', const=True, default=True)
ap.add_argument('--seed',               type=int,      default=42)
ap.add_argument('--output_dir',         type=str,      default='ruler_results')
ap.add_argument('--save_heatmap',       type=str2bool, nargs='?', const=True, default=True)
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

RECALL_K = [1, 5, 10]
MRR_K    = 10

# ── NIAH task configuration (matches RULER) ────────────────────────────────────

# task_name → (num_key_value_pairs, value_type)
# value_type: 'number' | 'word' | 'uuid'
NIAH_CONFIG = {
    'niah_single_1':   (1,  'number'),
    'niah_single_2':   (1,  'word'),
    'niah_single_3':   (1,  'uuid'),
    'niah_multikey_1': (4,  'number'),
    'niah_multikey_2': (8,  'number'),
    'niah_multikey_3': (16, 'number'),
    'niah_multivalue': (4,  'number'),   # 4 values for ONE key
    'niah_multiquery': (8,  'number'),   # 8 pairs; each example queries a different one
}

# RULER type_needle_v strings (used in needle sentence and query)
NEEDLE_TYPE_STR = {
    'number': 'number',
    'word':   'word',
    'uuid':   'UUID',
}

TASK_INSTRUCTIONS = {
    'niah_single_1':   'Given a question, retrieve the document containing the specific answer',
    'niah_single_2':   'Given a question, retrieve the document containing the specific answer',
    'niah_single_3':   'Given a question, retrieve the document containing the specific answer',
    'niah_multikey_1': 'Given a question about a specific key, retrieve the document containing its value',
    'niah_multikey_2': 'Given a question about a specific key, retrieve the document containing its value',
    'niah_multikey_3': 'Given a question about a specific key, retrieve the document containing its value',
    'niah_multivalue': 'Given a question, retrieve the document containing all the specified values',
    'niah_multiquery': 'Given a question, retrieve the document containing the specific answer',
    'cwe': ('Given a list of words that appear most frequently in a document, '
            'retrieve the document where these words are the most common'),
    'fwe': ('Given a list of coded words that appear most frequently in a document, '
            'retrieve the document where these coded words appear most often'),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_haystack(path: str) -> str:
    """Load Paul Graham essays from JSON produced by download_paulgraham_essay.py."""
    with open(path) as f:
        data = json.load(f)
    text = data['text']
    logger.info(f'Loaded haystack: {len(text):,} chars from {path}')
    return text


def load_english_words(path: str) -> List[str]:
    """Load english_words.json from RULER repo."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        words = data
    elif isinstance(data, dict):
        # Some versions are {"words": [...]} or a freq dict
        words = data.get('words', list(data.values()))
    else:
        raise ValueError(f'Unexpected english_words.json format: {type(data)}')
    # Keep only clean single-token-ish words, filter very short/long
    words = [w.strip().lower() for w in words if 3 <= len(w.strip()) <= 12
             and w.strip().isalpha()]
    logger.info(f'Loaded {len(words):,} English words from {path}')
    return words



# ── Haystack token pool (cached) ───────────────────────────────────────────────

_HAYSTACK_TOKEN_POOL: Optional[List[int]] = None

def _get_haystack_pool(tokenizer) -> List[int]:
    global _HAYSTACK_TOKEN_POOL
    if _HAYSTACK_TOKEN_POOL is None:
        text = _HAYSTACK_TEXT
        ids  = tokenizer.encode(text, add_special_tokens=False)
        # Repeat to ensure enough tokens for any context length
        target = max(args.lengths) * 4
        while len(ids) < target:
            ids = ids + ids
        _HAYSTACK_TOKEN_POOL = ids
        logger.info(f'Haystack pool: {len(ids):,} tokens')
    return _HAYSTACK_TOKEN_POOL


def _haystack_segment(tokenizer, length: int, seed_offset: int) -> List[int]:
    """Return `length` haystack tokens starting at a deterministic offset."""
    pool   = _get_haystack_pool(tokenizer)
    # Rotate by seed_offset to get different segments per example
    rotate = (seed_offset * 997) % max(1, len(pool) - length - 1)
    seg    = pool[rotate : rotate + length]
    if len(seg) < length:
        seg = (pool + pool)[rotate : rotate + length]
    return seg[:length]


def _inject_at_depth(haystack: List[int], needle: List[int], depth: float) -> List[int]:
    pos = int(len(haystack) * depth)
    return haystack[:pos] + needle + haystack[pos:]


# ── Value generators (matching RULER) ─────────────────────────────────────────

def _make_value(rng: random.Random, val_type: str,
                english_words: List[str]) -> str:
    if val_type == 'number':
        return str(rng.randint(1_000_000, 9_999_999))
    elif val_type == 'word':
        return rng.choice(english_words)
    elif val_type == 'uuid':
        # UUID seeded from rng state for reproducibility
        return str(_uuid.UUID(int=rng.getrandbits(128)))
    raise ValueError(val_type)


# ── NIAH document builder ──────────────────────────────────────────────────────

def build_niah_gold(tokenizer, rng: random.Random, context_len: int,
                    depth: float, task: str, english_words: List[str],
                    seed: int) -> Tuple[str, str, List[str]]:
    """
    Build one NIAH gold document.

    Needle format (RULER): "The special magic {type_needle_v} for {key} is {value}."
    Query format  (RULER): "What is/are the special magic {type_needle_v} for {key}?"
    """
    n_pairs, val_type  = NIAH_CONFIG[task]
    type_needle_v      = NEEDLE_TYPE_STR[val_type]
    is_multivalue      = (task == 'niah_multivalue')

    # For multivalue: 1 key, n_pairs values
    # For everything else: n_pairs distinct keys, 1 value each
    if is_multivalue:
        key  = rng.choice(english_words)
        keys = [key] * n_pairs
    else:
        keys = rng.sample(english_words, min(n_pairs, len(english_words)))

    values  = [_make_value(rng, val_type, english_words) for _ in range(n_pairs)]

    # ── Needle sentence(s) ────────────────────────────────────────────────────
    # RULER template: "The special magic {type_needle_v} for {key} is {value}."
    needles = [f"The special magic {type_needle_v} for {key} is {value}."
               for key, value in zip(keys, values)]
    needle_text = "  ".join(needles)
    needle_ids  = tokenizer.encode(" " + needle_text + " ", add_special_tokens=False)

    # ── Haystack + insertion ──────────────────────────────────────────────────
    haystack_len = max(0, context_len - len(needle_ids))
    haystack_ids = _haystack_segment(tokenizer, haystack_len, seed)
    doc_ids      = _inject_at_depth(haystack_ids, needle_ids, depth)[:context_len]
    doc_text     = tokenizer.decode(doc_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)

    # ── Query + gold answers ──────────────────────────────────────────────────
    if is_multivalue:
        # RULER: "What are all the special magic {type} for {key}?"
        query       = f"What are all the special magic {type_needle_v}s for {keys[0]}?"
        gold_answers = values
    elif task == 'niah_multiquery':
        # Each example independently queries a different key
        idx         = rng.randint(0, n_pairs - 1)
        query       = f"What is the special magic {type_needle_v} for {keys[idx]}?"
        gold_answers = [values[idx]]
    elif task.startswith('niah_multikey'):
        # Multiple keys in the doc; query picks one
        idx         = rng.randint(0, n_pairs - 1)
        query       = f"What is the special magic {type_needle_v} for {keys[idx]}?"
        gold_answers = [values[idx]]
    else:
        # niah_single_*: always key 0
        query       = f"What is the special magic {type_needle_v} for {keys[0]}?"
        gold_answers = [values[0]]

    return doc_text, query, gold_answers


def build_niah_distractor(tokenizer, context_len: int, seed: int) -> str:
    """Pure haystack, no needle."""
    ids = _haystack_segment(tokenizer, context_len, seed)
    return tokenizer.decode(ids[:context_len], skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)


# ── CWE document builder (RULER-aligned) ──────────────────────────────────────

def build_cwe_gold(tokenizer, rng: random.Random, context_len: int,
                   depth: float, english_words: List[str],
                   seed: int) -> Tuple[str, str, List[str]]:
    """
    CWE gold document (RULER-aligned):
      - Single numbered list of English words
      - `cwe_num_common` target words each appear `cwe_occurrences` times
      - First occurrence of any target word is at fractional position `depth`
      - Rest are distributed after that point
      - Format: "1. apple\n2. bird\n3. apple\n4. cat\n..."
    """
    common_words = rng.sample(english_words, args.cwe_num_common)
    common_set   = set(common_words)
    non_common   = [w for w in english_words if w not in common_set]

    # Estimate total list items: each "N. word\n" is ~3-5 tokens
    n_items_est = context_len // 4

    # Background word count (before and after target injection)
    n_target_total = args.cwe_num_common * args.cwe_occurrences
    n_base         = max(n_items_est - n_target_total, n_items_est // 2)

    base_words  = rng.choices(non_common, k=n_base)
    first_target_idx = max(0, int(n_base * depth))

    # Words before first occurrence
    words_before = base_words[:first_target_idx]

    # Insert first occurrence of each common word at depth point
    first_occ = list(common_words)
    rng.shuffle(first_occ)

    # Remaining occurrences (occurrences-1 of each) distributed after
    remaining = common_words * (args.cwe_occurrences - 1) + base_words[first_target_idx:]
    rng.shuffle(remaining)

    all_words = words_before + first_occ + remaining

    # Format as numbered list (RULER format: "1. word\n2. word\n...")
    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(all_words))
    ids      = tokenizer.encode(numbered, add_special_tokens=False)[:context_len]
    doc_text = tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)

    word_list = ", ".join(common_words)
    query     = (f"Find the document where these words appear most frequently "
                 f"in the numbered word list: {word_list}")
    return doc_text, query, common_words


def build_cwe_distractor(tokenizer, rng: random.Random,
                          english_words: List[str], context_len: int,
                          seed: int) -> str:
    """CWE distractor: numbered list with uniform word frequencies."""
    drng     = random.Random(seed)
    n_items  = context_len // 4
    words    = drng.choices(english_words, k=n_items)
    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(words))
    ids      = tokenizer.encode(numbered, add_special_tokens=False)[:context_len]
    return tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)


# ── FWE document builder (RULER-aligned) ──────────────────────────────────────

def _coded_word(rng: random.Random, length: int = None) -> str:
    """Random lowercase letter sequence (RULER-style coded word)."""
    n = length or rng.randint(3, 6)
    return ''.join(rng.choices(string.ascii_lowercase, k=n))


def build_fwe_gold(tokenizer, rng: random.Random, context_len: int,
                   depth: float, seed: int) -> Tuple[str, str, List[str]]:
    """
    FWE gold document (RULER-aligned):
      - Coded text: "word1....word2....word3...." (dots are RULER's separator)
      - `fwe_num_targets` coded words each appear `fwe_frequency` times
      - First occurrence of any target coded word at fractional position `depth`
      - Rest distributed after
      - Coded vocab: random letter sequences NOT in target set
    """
    # Generate unique target coded words
    target_words = []
    used = set()
    while len(target_words) < args.fwe_num_targets:
        w = _coded_word(rng, length=5)   # fixed length 5 for uniqueness
        if w not in used:
            target_words.append(w)
            used.add(w)

    # Background coded vocabulary
    vocab_size = 300
    coded_vocab = []
    while len(coded_vocab) < vocab_size:
        w = _coded_word(rng)
        if w not in used:
            coded_vocab.append(w)
            used.add(w)

    # Estimate items: each "word...." is ~6-10 tokens
    n_items_est    = context_len // 7
    n_target_total = args.fwe_num_targets * args.fwe_frequency
    n_base         = max(n_items_est - n_target_total, n_items_est // 2)

    base_coded       = rng.choices(coded_vocab, k=n_base)
    first_target_idx = max(0, int(n_base * depth))

    words_before = base_coded[:first_target_idx]
    first_occ    = list(target_words)
    rng.shuffle(first_occ)

    remaining = (target_words * (args.fwe_frequency - 1)) + base_coded[first_target_idx:]
    rng.shuffle(remaining)

    all_words = words_before + first_occ + remaining

    # RULER FWE format: dots between coded words
    coded_text = "....".join(all_words)
    ids        = tokenizer.encode(coded_text, add_special_tokens=False)[:context_len]
    doc_text   = tokenizer.decode(ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)

    word_list = ", ".join(target_words)
    query     = (f"Find the document where these coded words appear most frequently "
                 f"in the coded text: {word_list}")
    return doc_text, query, target_words


def build_fwe_distractor(tokenizer, rng: random.Random,
                          context_len: int, seed: int) -> str:
    """FWE distractor: coded text with no dominant word."""
    drng       = random.Random(seed)
    vocab      = [_coded_word(drng) for _ in range(300)]
    n_items    = context_len // 7
    words      = drng.choices(vocab, k=n_items)
    coded_text = "....".join(words)
    ids        = tokenizer.encode(coded_text, add_special_tokens=False)[:context_len]
    return tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)


# ── Embedding model ────────────────────────────────────────────────────────────

def merge_peft_if_needed() -> Tuple[str, bool]:
    if not args.peft:
        return args.model, False
    assert args.base_model
    tmp = os.path.join(args.output_dir, f'_merged_{str(_uuid.uuid4())[:8]}')
    os.makedirs(tmp, exist_ok=True)
    tok  = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model,
               torch_dtype=torch.bfloat16, device_map='auto')
    m    = PeftModel.from_pretrained(base, args.model).merge_and_unload()
    m.save_pretrained(tmp); tok.save_pretrained(tmp)
    return tmp, True


def _insert_lmk(ids: List[int], gran: int, eos_id: int, max_len: int) -> List[int]:
    result = []
    for i, tok in enumerate(ids):
        if i % gran == 0:
            result.append(eos_id)
            if len(result) >= max_len: break
        result.append(tok)
        if len(result) >= max_len: break
    return result[:max_len]


class DecoderEmbedder:
    def __init__(self, model_path: str, doc_max_len: int):
        self.pool     = args.pool
        self.gran     = args.lmk_granularity
        self.doc_max  = doc_max_len
        self.qry_max  = min(doc_max_len, 512)
        self.add_eos  = (args.pool != 'lmk')
        self.norm     = args.normalize

        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            # Use unk, NOT eos — eos is the LMK landmark; using it as pad
            # causes attention_mask to zero-out all landmark positions
            if self.tok.unk_token:
                self.tok.pad_token    = self.tok.unk_token
                self.tok.pad_token_id = self.tok.unk_token_id
            else:
                self.tok.add_special_tokens({'pad_token': '<pad>'})
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id

        wm = models.Transformer(
            model_path, max_seq_length=doc_max_len,
            config_args={'trust_remote_code': True},
            model_args={'trust_remote_code': True, 'torch_dtype': torch.bfloat16, "attn_implementation": "flash_attention_2",})
        self._st = SentenceTransformer(modules=[wm])
        self._st.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Embedder | pool={self.pool} gran={self.gran} '
                    f'doc_max={doc_max_len} | {self.device}')

    def _tokenize(self, texts: List[str], is_doc: bool) -> Dict[str, torch.Tensor]:
        max_out = self.doc_max if is_doc else self.qry_max
        if self.pool == 'lmk':
            budget  = max_out - (max_out // self.gran) - 1
            raw     = self.tok(texts, padding=False, truncation=True, max_length=budget,
                               add_special_tokens=True, return_attention_mask=False)
            new_ids = [_insert_lmk(ids, self.gran, self.eos_id, max_out)
                       for ids in raw['input_ids']]
        else:
            budget  = max_out - (1 if self.add_eos else 0)
            raw     = self.tok(texts, padding=False, truncation=True, max_length=budget,
                               add_special_tokens=True, return_attention_mask=False)
            new_ids = ([ids + [self.eos_id] for ids in raw['input_ids']] if self.add_eos
                       else raw['input_ids'])

        left   = (self.pool == 'last')
        maxlen = max(len(ids) for ids in new_ids)
        padded = [([self.pad_id]*(maxlen-len(ids)) + ids) if left
                  else (ids + [self.pad_id]*(maxlen-len(ids)))
                  for ids in new_ids]
        inp  = torch.tensor(padded, dtype=torch.long)
        mask = (inp != self.pad_id).long()
        return {'input_ids': inp, 'attention_mask': mask}

    @torch.no_grad()
    def _forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = {k: v.to(self.device) for k, v in feats.items()}
        out   = self._st._first_module().auto_model(
            **{k: feats[k] for k in ['input_ids','attention_mask']},
            return_dict=True)
        h, ids, mask = out.last_hidden_state, feats['input_ids'], feats['attention_mask']

        if self.pool == 'last':
            emb = (h[:,-1] if mask[:,-1].all() else
                   h[torch.arange(h.size(0)), (mask.sum(1)-1).clamp(0)])
        elif self.pool == 'mean':
            mf  = mask.unsqueeze(-1).to(h.dtype)
            emb = (h * mf).sum(1) / mf.sum(1).clamp_min(1e-9)
        else:  # lmk
            lm  = (ids == self.eos_id) & mask.bool()
            lmf = lm.to(h.dtype).unsqueeze(-1)
            emb = (h * lmf).sum(1) / lm.sum(1).clamp_min(1).unsqueeze(-1).to(h.dtype)
            no  = lm.sum(1) == 0
            if no.any():
                mf  = mask.unsqueeze(-1).to(h.dtype)
                emb[no] = ((h * mf).sum(1) / mf.sum(1).clamp_min(1e-9))[no]

        if self.norm:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.float().cpu()

    def encode(self, texts: List[str], is_doc: bool, desc: str = '') -> np.ndarray:
        out, bs = [], args.batch_size
        it = (tqdm(range(0, len(texts), bs), desc=desc, leave=False)
              if desc else range(0, len(texts), bs))
        for i in it:
            out.append(self._forward(self._tokenize(texts[i:i+bs], is_doc)).numpy())
        return np.concatenate(out, axis=0)


# ── Distractor cache (one set per (task, context_len)) ────────────────────────
_DISTRACTOR_CACHE: Dict[Tuple[str,int], np.ndarray] = {}

def get_distractors(embedder: DecoderEmbedder, task: str,
                    context_len: int, english_words: List[str]) -> np.ndarray:
    key = (task, context_len)
    if key in _DISTRACTOR_CACHE:
        return _DISTRACTOR_CACHE[key]

    n   = args.corpus_size - 1
    rng = random.Random(args.seed ^ hash(key) & 0xFFFF)
    logger.info(f'Encoding {n} distractors | {task} | {context_len//1024}k')

    docs = []
    for i in range(n):
        s = args.seed + i * 9973 + context_len
        if task.startswith('niah'):
            docs.append(build_niah_distractor(embedder.tok, context_len, s))
        elif task == 'cwe':
            docs.append(build_cwe_distractor(embedder.tok, rng, english_words,
                                              context_len, s))
        elif task == 'fwe':
            docs.append(build_fwe_distractor(embedder.tok, rng, context_len, s))

    embs = embedder.encode(docs, is_doc=True,
                           desc=f'distractors {task}@{context_len//1024}k')
    _DISTRACTOR_CACHE[key] = embs
    return embs


# ── Metrics ────────────────────────────────────────────────────────────────────

def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / len(set(gold)) if gold else 0.0

def mrr_at_k(ranked, gold, k):
    gs = set(gold)
    for rank, idx in enumerate(ranked[:k], 1):
        if idx in gs: return 1.0 / rank
    return 0.0


# ── Cell evaluation ────────────────────────────────────────────────────────────

def evaluate_cell(embedder: DecoderEmbedder, task: str, context_len: int,
                  depth: float, english_words: List[str]) -> Dict:
    rng          = random.Random(args.seed ^ hash((task, context_len, depth)) & 0xFFFF)
    instruction  = TASK_INSTRUCTIONS.get(task, '')
    dist_embs    = get_distractors(embedder, task, context_len, english_words)
    acc          = defaultdict(list)

    for ex_idx in tqdm(range(args.num_examples),
                       desc=f'{task}@{context_len//1024}k depth={depth:.0%}',
                       leave=False):
        s = args.seed + ex_idx * 1009 + int(depth * 100) + context_len

        # Build gold document
        if task.startswith('niah'):
            gold_text, query, gold_ans = build_niah_gold(
                embedder.tok, rng, context_len, depth, task, english_words, s)
        elif task == 'cwe':
            gold_text, query, gold_ans = build_cwe_gold(
                embedder.tok, rng, context_len, depth, english_words, s)
        elif task == 'fwe':
            gold_text, query, gold_ans = build_fwe_gold(
                embedder.tok, rng, context_len, depth, s)
        else:
            raise ValueError(task)

        # Encode gold
        gold_emb = embedder.encode([gold_text], is_doc=True)  # (1, D)

        # Build corpus: gold at index 0, then distractors; shuffle
        corpus = np.concatenate([gold_emb, dist_embs], axis=0)
        perm   = list(range(len(corpus)))
        rng.shuffle(perm)
        corpus = corpus[perm]
        gold_in_corpus = [perm.index(0)]

        # Encode query
        q_text = (f'Instruct: {instruction}\nQuery: {query}'
                  if args.add_prompt else query)
        q_emb  = embedder.encode([q_text], is_doc=False)

        # Rank
        sims   = (q_emb @ corpus.T).squeeze(0)
        ranked = sims.argsort()[::-1].tolist()

        for k in RECALL_K:
            acc[f'recall@{k}'].append(recall_at_k(ranked, gold_in_corpus, k))
        acc[f'mrr@{MRR_K}'].append(mrr_at_k(ranked, gold_in_corpus, MRR_K))

    result = {k: float(np.mean(v)) for k, v in acc.items()}
    result['n'] = args.num_examples
    logger.info(
        f'  {task:<16} {context_len//1024:>5}k  depth={depth:.0%}  '
        f"R@1={result['recall@1']:.3f}  R@5={result['recall@5']:.3f} R@10={result['recall@10']:.3f} "
        # f"MRR@10={result.get(f'mrr@{MRR_K}',0):.3f}"
    )
    return result


# ── Output ──────────────────────────────────────────────────────────────────────

def save_outputs(all_results: Dict, out_dir: Path, run_tag: str):
    tasks   = sorted({t for t,_,_ in all_results})
    lengths = sorted({l for _,l,_ in all_results})
    depths  = sorted({d for _,_,d in all_results})

    # Summary CSV
    with (out_dir / f'{run_tag}_summary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task','length_k','depth','recall@1','recall@5',
                    'recall@10',f'mrr@{MRR_K}','n'])
        for (task, length, depth), res in sorted(all_results.items()):
            w.writerow([task, length//1024, f'{depth:.0%}',
                        f"{res.get('recall@1',0):.4f}",
                        f"{res.get('recall@5',0):.4f}",
                        f"{res.get('recall@10',0):.4f}",
                        f"{res.get(f'mrr@{MRR_K}',0):.4f}",
                        res.get('n','')])

    # Per-task grid CSVs + heatmaps
    depths_rev = list(reversed(depths))   # top row = depth 100% (end of doc)
    for task in tasks:
        for metric in [f'recall@{k}' for k in RECALL_K] + [f'mrr@{MRR_K}']:
            safe = metric.replace('@','_at_')
            # Grid CSV: rows=depth, cols=length
            with (out_dir / f'{run_tag}_{task}_{safe}_grid.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['depth\\length'] + [f'{l//1024}k' for l in lengths])
                for depth in depths:
                    row = [f'{depth:.0%}']
                    for length in lengths:
                        v = all_results.get((task, length, depth), {}).get(metric)
                        row.append(f'{v:.4f}' if v is not None else '')
                    w.writerow(row)

        # Heatmap
        if args.save_heatmap:
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            except ImportError:
                continue

            metrics_plot = [f'recall@{k}' for k in RECALL_K] + [f'mrr@{MRR_K}']
            fig, axes = plt.subplots(1, len(metrics_plot),
                                     figsize=(4*len(metrics_plot), 2+len(depths)*0.6))
            if len(metrics_plot) == 1: axes = [axes]

            for ax, metric in zip(axes, metrics_plot):
                grid = np.full((len(depths_rev), len(lengths)), np.nan)
                for i, d in enumerate(depths_rev):
                    for j, l in enumerate(lengths):
                        v = all_results.get((task, l, d), {}).get(metric)
                        if v is not None: grid[i,j] = v

                im = ax.imshow(grid, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(metric, fontsize=8)
                ax.set_xticks(range(len(lengths)))
                ax.set_xticklabels([f'{l//1024}k' for l in lengths], fontsize=7)
                ax.set_yticks(range(len(depths_rev)))
                ax.set_yticklabels([f'{d:.0%}' for d in depths_rev], fontsize=7)
                ax.set_xlabel('Context Length', fontsize=7)
                ax.set_ylabel('Needle Depth\n(0%=start, 100%=end)', fontsize=7)
                for i in range(len(depths_rev)):
                    for j in range(len(lengths)):
                        if not np.isnan(grid[i,j]):
                            c = 'black' if 0.2 < grid[i,j] < 0.8 else 'white'
                            ax.text(j, i, f'{grid[i,j]:.2f}', ha='center',
                                    va='center', fontsize=6, color=c, fontweight='bold')

            fig.suptitle(f'{task}  |  {run_tag}', fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f'{run_tag}_{task}_heatmap.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    # JSON dump
    with (out_dir / f'{run_tag}_all.json').open('w') as f:
        json.dump({f'{t}|{l}|{d}': v for (t,l,d),v in all_results.items()},
                  f, indent=2)

    # Console grid per task (recall@1)
    for task in tasks:
        print(f'\n{"="*65}\n {task}  |  Recall@1')
        print(f' rows=depth (0%=doc_start, 100%=doc_end)  cols=context_len')
        print('='*65)
        print(f'{"depth":>8}' + ''.join(f'  {l//1024:>4}k' for l in lengths) + '   avg')
        print('-'*65)
        for depth in depths:
            vals, row = [], f'{depth:>7.0%}'
            for length in lengths:
                v = all_results.get((task, length, depth), {}).get('recall@1')
                row += f'  {v:.3f}' if v is not None else '   --'
                if v is not None: vals.append(v)
            row += f'  {np.mean(vals):.3f}' if vals else '   --'
            print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

# Module-level globals (populated in main before any data generation)
_HAYSTACK_TEXT:   str       = ''
_ENGLISH_WORDS:   List[str] = []


def main():
    global _HAYSTACK_TEXT, _ENGLISH_WORDS

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load RULER data sources
    _HAYSTACK_TEXT = load_haystack(args.haystack_file)
    _ENGLISH_WORDS = load_english_words(args.english_words_file)

    # Warm haystack token pool for largest length
    model_path, is_temp = merge_peft_if_needed()
    doc_max_len = max(args.lengths)
    embedder    = DecoderEmbedder(model_path, doc_max_len=doc_max_len)
    if is_temp:
        shutil.rmtree(model_path); logger.info('Cleaned up temp merged model')
    for layer in embedder._st._first_module().auto_model.layers:
        layer.self_attn.is_causal = False
        # FA2 also uses this in _flash_attention_forward
        if hasattr(layer.self_attn, '_flash_attention_forward'):
            layer.self_attn.config._attn_implementation  # just verify
    layer = embedder._st._first_module().auto_model.layers[0]
    print('-'*100)
    print(type(layer.self_attn))   # should show FlashAttention or similar
    print(layer.self_attn.config._attn_implementation)  # should be "flash_attention_2"

    # Pre-warm haystack pool
    _get_haystack_pool(embedder.tok)

    total = len(args.tasks) * len(args.lengths) * len(args.depths)
    logger.info(
        f'Grid: {len(args.tasks)} tasks × {len(args.lengths)} lengths × '
        f'{len(args.depths)} depths = {total} cells  |  '
        f'{args.num_examples} examples × corpus={args.corpus_size}'
    )

    all_results: Dict = {}
    for task in args.tasks:
        for context_len in args.lengths:
            for depth in args.depths:
                all_results[(task, context_len, depth)] = evaluate_cell(
                    embedder, task, context_len, depth, _ENGLISH_WORDS)

    pool_tag  = args.pool + (f'_gran{args.lmk_granularity}' if args.pool == 'lmk' else '')
    model_tag = '_'.join(args.model.rstrip('/').split('/')[-2:])
    run_tag   = f'{pool_tag}_corp{args.corpus_size}_{model_tag}'

    save_outputs(all_results, out_dir, run_tag)


if __name__ == '__main__':
    main()