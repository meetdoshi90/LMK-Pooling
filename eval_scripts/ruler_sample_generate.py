#!/usr/bin/env python3
"""
generate_samples.py  —  Generate human-readable sample corpus + query files
for each RULER-aligned NIAH/CWE/FWE task WITHOUT running any model.

Output structure:
  samples/
    <task>/
      example_001/
        QUERY.txt          ← the query, with KEY LOOKUP annotation
        doc_00_GOLD.txt    ← the gold document (contains the needle)
        doc_01_distractor.txt
        ...
        META.txt           ← needle text, gold answers, depth, context_len

Usage:
  python generate_samples.py \\
      --haystack_file PaulGrahamEssays.json \\
      --english_words_file english_words.json \\
      --lengths 512 1024 \\
      --depths 0.0 0.5 1.0 \\
      --num_examples 2 \\
      --corpus_size 5 \\
      --output_dir samples
"""

import os, re, json, math, random, string, argparse, uuid as _uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── CLI ────────────────────────────────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','1'): return True
    if v.lower() in ('no','false','f','0'): return False
    raise argparse.ArgumentTypeError('Boolean expected.')

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--haystack_file',      type=str, default='PaulGrahamEssays.json')
ap.add_argument('--english_words_file', type=str, default='english_words.json')
ap.add_argument('--lengths',            type=int, nargs='+', default=[512, 1024])
ap.add_argument('--depths',             type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
ap.add_argument('--tasks',              type=str, nargs='+',
                default=['niah_single_1','niah_single_2','niah_single_3',
                         'niah_multikey_1','niah_multikey_2','niah_multikey_3',
                         'niah_multivalue','niah_multiquery','cwe','fwe'])
ap.add_argument('--num_examples',       type=int, default=2,
                help='Examples per (task, length, depth) cell')
ap.add_argument('--corpus_size',        type=int, default=5,
                help='Documents per corpus (1 gold + corpus_size-1 distractors)')
ap.add_argument('--cwe_num_common',     type=int, default=3)
ap.add_argument('--cwe_occurrences',    type=int, default=10)
ap.add_argument('--fwe_num_targets',    type=int, default=3)
ap.add_argument('--fwe_frequency',      type=int, default=15)
ap.add_argument('--seed',               type=int, default=42)
ap.add_argument('--output_dir',         type=str, default='samples')
args = ap.parse_args()

random.seed(args.seed)

# ── Task config ────────────────────────────────────────────────────────────────

NIAH_CONFIG = {
    'niah_single_1':   (1,  'number'),
    'niah_single_2':   (1,  'word'),
    'niah_single_3':   (1,  'uuid'),
    'niah_multikey_1': (4,  'number'),
    'niah_multikey_2': (8,  'number'),
    'niah_multikey_3': (16, 'number'),
    'niah_multivalue': (4,  'number'),
    'niah_multiquery': (8,  'number'),
}
NEEDLE_TYPE_STR = {'number': 'number', 'word': 'word', 'uuid': 'UUID'}

TASK_DESCRIPTIONS = {
    'niah_single_1':   'Single needle (7-digit number value). The model must find the one document containing a specific key→value pair.',
    'niah_single_2':   'Single needle (English word value). The model must find the one document containing a specific key→value pair.',
    'niah_single_3':   'Single needle (UUID value). The model must find the one document containing a specific key→value pair.',
    'niah_multikey_1': '4 distinct key→value pairs hidden in the document; query asks for one of them.',
    'niah_multikey_2': '8 distinct key→value pairs hidden in the document; query asks for one of them.',
    'niah_multikey_3': '16 distinct key→value pairs hidden in the document; query asks for one of them.',
    'niah_multivalue': '4 different values for the SAME key hidden in the document; query asks for ALL values.',
    'niah_multiquery': '8 distinct key→value pairs in the document; each example queries a different one.',
    'cwe':             'Common Words Extraction. A numbered word list where target words appear N times. Query names those words; model must find the document where they are most frequent.',
    'fwe':             'Frequent Words Extraction. A coded text where target coded-words appear N times. Query names those coded-words; model must find the document where they appear most.',
}

# ── Data loading (character-level, no tokenizer needed) ───────────────────────

def load_haystack(path: str) -> str:
    with open(path) as f:
        data = json.load(f)
    text = data['text'] if isinstance(data, dict) else '\n'.join(data)
    print(f'Loaded haystack: {len(text):,} chars from {path}')
    return text

def load_english_words(path: str) -> List[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        words = data
    elif isinstance(data, dict):
        words = data.get('words', list(data.values()))
    else:
        raise ValueError(f'Unexpected english_words.json format: {type(data)}')
    words = [w.strip().lower() for w in words
             if 3 <= len(w.strip()) <= 12 and w.strip().isalpha()]
    print(f'Loaded {len(words):,} English words from {path}')
    return words

# ── Haystack helpers (char-level approximation) ───────────────────────────────
# We use ~4 chars per token as a rough approximation so we don't need a tokenizer.

CHARS_PER_TOK = 4

def _haystack_segment(text: str, n_chars: int, seed_offset: int) -> str:
    """Extract n_chars from the haystack starting at a seeded offset."""
    rotate = (seed_offset * 997) % max(1, len(text) - n_chars - 1)
    seg = text[rotate: rotate + n_chars]
    if len(seg) < n_chars:
        seg = (text + text)[rotate: rotate + n_chars]
    return seg[:n_chars]

def _inject_at_depth(haystack: str, needle: str, depth: float) -> str:
    pos = int(len(haystack) * depth)
    # Snap to nearest sentence/word boundary to avoid mid-word insertion
    snap = haystack.rfind(' ', 0, pos)
    pos = snap if snap > 0 else pos
    return haystack[:pos] + ' ' + needle + ' ' + haystack[pos:]

# ── Value generators ───────────────────────────────────────────────────────────

def _make_value(rng: random.Random, val_type: str, english_words: List[str]) -> str:
    if val_type == 'number':
        return str(rng.randint(1_000_000, 9_999_999))
    elif val_type == 'word':
        return rng.choice(english_words)
    elif val_type == 'uuid':
        return str(_uuid.UUID(int=rng.getrandbits(128)))
    raise ValueError(val_type)

# ── NIAH builders ──────────────────────────────────────────────────────────────

def build_niah_gold(haystack: str, rng: random.Random, context_len: int,
                    depth: float, task: str, english_words: List[str],
                    seed: int) -> Tuple[str, str, List[str], str, List[str]]:
    """Returns (doc_text, query, gold_answers, needle_text, all_keys)."""
    n_pairs, val_type = NIAH_CONFIG[task]
    type_needle_v     = NEEDLE_TYPE_STR[val_type]
    is_multivalue     = (task == 'niah_multivalue')

    if is_multivalue:
        key  = rng.choice(english_words)
        keys = [key] * n_pairs
    else:
        keys = rng.sample(english_words, min(n_pairs, len(english_words)))

    values  = [_make_value(rng, val_type, english_words) for _ in range(n_pairs)]
    needles = [f"The special magic {type_needle_v} for {k} is {v}."
               for k, v in zip(keys, values)]
    needle_text = '  '.join(needles)

    n_chars  = context_len * CHARS_PER_TOK
    haystack_chars = max(0, n_chars - len(needle_text) - 2)
    hs       = _haystack_segment(haystack, haystack_chars, seed)
    doc_text = _inject_at_depth(hs, needle_text, depth)[:n_chars]

    if is_multivalue:
        query        = f"What are all the special magic {type_needle_v}s for {keys[0]}?"
        gold_answers = values
    elif task == 'niah_multiquery':
        idx          = rng.randint(0, n_pairs - 1)
        query        = f"What is the special magic {type_needle_v} for {keys[idx]}?"
        gold_answers = [values[idx]]
    elif task.startswith('niah_multikey'):
        idx          = rng.randint(0, n_pairs - 1)
        query        = f"What is the special magic {type_needle_v} for {keys[idx]}?"
        gold_answers = [values[idx]]
    else:
        query        = f"What is the special magic {type_needle_v} for {keys[0]}?"
        gold_answers = [values[0]]

    return doc_text, query, gold_answers, needle_text, keys

def build_niah_distractor(haystack: str, context_len: int, seed: int) -> str:
    n_chars = context_len * CHARS_PER_TOK
    return _haystack_segment(haystack, n_chars, seed)

# ── CWE builders ──────────────────────────────────────────────────────────────

def build_cwe_gold(english_words: List[str], rng: random.Random,
                   context_len: int, depth: float,
                   seed: int) -> Tuple[str, str, List[str]]:
    common_words = rng.sample(english_words, args.cwe_num_common)
    common_set   = set(common_words)
    non_common   = [w for w in english_words if w not in common_set]

    n_items_est    = (context_len * CHARS_PER_TOK) // 10  # ~10 chars per "N. word\n"
    n_target_total = args.cwe_num_common * args.cwe_occurrences
    n_base         = max(n_items_est - n_target_total, n_items_est // 2)

    base_words       = rng.choices(non_common, k=n_base)
    first_target_idx = max(0, int(n_base * depth))
    words_before     = base_words[:first_target_idx]
    first_occ        = list(common_words); rng.shuffle(first_occ)
    remaining        = common_words * (args.cwe_occurrences - 1) + base_words[first_target_idx:]
    rng.shuffle(remaining)

    all_words = words_before + first_occ + remaining
    numbered  = '\n'.join(f'{i+1}. {w}' for i, w in enumerate(all_words))
    doc_text  = numbered[: context_len * CHARS_PER_TOK]

    word_list = ', '.join(common_words)
    query     = (f"Find the document where these words appear most frequently "
                 f"in the numbered word list: {word_list}")
    return doc_text, query, common_words

def build_cwe_distractor(english_words: List[str], rng: random.Random,
                          context_len: int, seed: int) -> str:
    drng    = random.Random(seed)
    n_items = (context_len * CHARS_PER_TOK) // 10
    words   = drng.choices(english_words, k=n_items)
    return '\n'.join(f'{i+1}. {w}' for i, w in enumerate(words))[: context_len * CHARS_PER_TOK]

# ── FWE builders ──────────────────────────────────────────────────────────────

def _coded_word(rng: random.Random, length: int = None) -> str:
    n = length or rng.randint(3, 6)
    return ''.join(rng.choices(string.ascii_lowercase, k=n))

def build_fwe_gold(rng: random.Random, context_len: int,
                   depth: float, seed: int) -> Tuple[str, str, List[str]]:
    target_words, used = [], set()
    while len(target_words) < args.fwe_num_targets:
        w = _coded_word(rng, length=5)
        if w not in used:
            target_words.append(w); used.add(w)

    coded_vocab = []
    while len(coded_vocab) < 300:
        w = _coded_word(rng)
        if w not in used:
            coded_vocab.append(w); used.add(w)

    n_items_est    = (context_len * CHARS_PER_TOK) // 9
    n_target_total = args.fwe_num_targets * args.fwe_frequency
    n_base         = max(n_items_est - n_target_total, n_items_est // 2)

    base_coded       = rng.choices(coded_vocab, k=n_base)
    first_target_idx = max(0, int(n_base * depth))
    words_before     = base_coded[:first_target_idx]
    first_occ        = list(target_words); rng.shuffle(first_occ)
    remaining        = (target_words * (args.fwe_frequency - 1)) + base_coded[first_target_idx:]
    rng.shuffle(remaining)

    all_words  = words_before + first_occ + remaining
    coded_text = '....'.join(all_words)[: context_len * CHARS_PER_TOK]

    word_list = ', '.join(target_words)
    query     = (f"Find the document where these coded words appear most frequently "
                 f"in the coded text: {word_list}")
    return coded_text, query, target_words

def build_fwe_distractor(rng: random.Random, context_len: int, seed: int) -> str:
    drng     = random.Random(seed)
    vocab    = [_coded_word(drng) for _ in range(300)]
    n_items  = (context_len * CHARS_PER_TOK) // 9
    words    = drng.choices(vocab, k=n_items)
    return '....'.join(words)[: context_len * CHARS_PER_TOK]

# ── File writers ───────────────────────────────────────────────────────────────

DIVIDER = '─' * 80

def _key_lookup_annotation(task: str, query: str, gold_answers: List[str],
                            keys: Optional[List[str]] = None) -> str:
    """Human-readable annotation explaining what the retrieval model must find."""
    lines = [
        '╔══════════════════════════════════════════════════════════════════════════════╗',
        '║                          QUERY FILE                                         ║',
        '╚══════════════════════════════════════════════════════════════════════════════╝',
        '',
        f'TASK          : {task}',
        f'DESCRIPTION   : {TASK_DESCRIPTIONS[task]}',
        '',
        DIVIDER,
        'QUERY TEXT (what gets sent to the retrieval model)',
        DIVIDER,
        query,
        '',
        DIVIDER,
        'WHAT THE MODEL IS LOOKING FOR',
        DIVIDER,
    ]

    if task.startswith('niah'):
        _, val_type  = NIAH_CONFIG[task]
        type_str     = NEEDLE_TYPE_STR[val_type]
        is_multivalue = (task == 'niah_multivalue')

        # Extract target key from query
        m = re.search(r'for (\S+)\?', query)
        target_key = m.group(1) if m else '(see query)'

        lines += [
            f'  • Needle format : "The special magic {type_str} for <key> is <value>."',
            f'  • Target key    : {target_key}',
            f'  • Gold answer(s): {", ".join(gold_answers)}',
        ]
        if is_multivalue:
            lines += [
                f'  • Strategy      : All {len(gold_answers)} values for key "{target_key}" must appear in the gold doc.',
            ]
        elif keys and len(keys) > 1:
            lines += [
                f'  • All keys in doc: {", ".join(set(keys))}',
                f'  • NOTE: Only ONE key is queried; model must still retrieve the doc containing ALL {len(set(keys))} needles.',
            ]
        else:
            lines += [
                '  • Strategy: Find the document containing the needle sentence for this key.',
            ]

    elif task == 'cwe':
        lines += [
            f'  • Target words  : {", ".join(gold_answers)}',
            f'  • Each target word appears {args.cwe_occurrences}× in the gold document.',
            f'  • Distractor docs have the same format but NO elevated word frequency.',
            f'  • Strategy: Count word frequency across docs; retrieve the one with the highest count for these words.',
        ]

    elif task == 'fwe':
        lines += [
            f'  • Target coded-words : {", ".join(gold_answers)}',
            f'  • Each target coded-word appears {args.fwe_frequency}× in the gold document.',
            f'  • Distractor docs use a different frequent coded-word (not in the query).',
            f'  • Strategy: Count coded-word frequency; retrieve the doc where query words appear most.',
        ]

    return '\n'.join(lines)


def write_example(out_dir: Path, ex_idx: int, task: str,
                  context_len: int, depth: float,
                  gold_doc: str, distractors: List[str],
                  query: str, gold_answers: List[str],
                  needle_text: Optional[str],
                  keys: Optional[List[str]],
                  gold_pos: int):
    """Write one example to disk."""
    ex_dir = out_dir / f'len{context_len}_depth{depth:.2f}' / f'example_{ex_idx:03d}'
    ex_dir.mkdir(parents=True, exist_ok=True)

    # ── QUERY.txt ─────────────────────────────────────────────────────────────
    (ex_dir / 'QUERY.txt').write_text(
        _key_lookup_annotation(task, query, gold_answers, keys),
        encoding='utf-8')

    # ── META.txt ──────────────────────────────────────────────────────────────
    meta_lines = [
        f'task          : {task}',
        f'context_len   : {context_len} tokens (~{context_len * CHARS_PER_TOK} chars)',
        f'depth         : {depth:.0%}  (needle inserted at {depth:.0%} through the haystack)',
        f'gold_doc_pos  : doc_{gold_pos:02d}  (0-indexed position in shuffled corpus)',
        f'gold_answers  : {", ".join(gold_answers)}',
    ]
    if needle_text:
        meta_lines += ['', 'NEEDLE TEXT (exact string injected into gold doc)', DIVIDER, needle_text]
    if keys:
        meta_lines += ['', f'ALL KEYS in gold doc : {", ".join(set(keys))}']

    (ex_dir / 'META.txt').write_text('\n'.join(meta_lines), encoding='utf-8')

    # ── Document files ────────────────────────────────────────────────────────
    # Shuffle gold + distractors so gold isn't always first
    corpus = [(gold_doc, True)] + [(d, False) for d in distractors]
    rng    = random.Random(args.seed ^ ex_idx)
    rng.shuffle(corpus)

    for i, (doc, is_gold) in enumerate(corpus):
        label    = 'GOLD' if is_gold else 'distractor'
        filename = f'doc_{i:02d}_{label}.txt'
        header   = (
            f'{"=" * 80}\n'
            f'DOC {i:02d} | {"★ GOLD DOCUMENT ★" if is_gold else "distractor"}\n'
            f'task={task}  len={context_len}  depth={depth:.0%}\n'
        )
        if is_gold:
            header += f'Contains needle: {needle_text if needle_text else "(see query)"}\n'
        header += '=' * 80 + '\n\n'
        (ex_dir / filename).write_text(header + doc, encoding='utf-8')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    haystack      = load_haystack(args.haystack_file)
    english_words = load_english_words(args.english_words_file)

    # Write a top-level README
    readme_lines = [
        'RULER-aligned NIAH/CWE/FWE  —  Sample Corpus + Query Files',
        '=' * 60,
        '',
        'Directory structure:',
        '  <task>/',
        '    len<N>_depth<D>/',
        '      example_XXX/',
        '        QUERY.txt          ← query text + key-lookup annotation',
        '        META.txt           ← needle text, gold answers, depth info',
        '        doc_00_GOLD.txt    ← gold document (contains the needle)',
        '        doc_01_distractor.txt',
        '        ...',
        '',
        'Tasks in this sample:',
    ]
    for t in args.tasks:
        readme_lines.append(f'  {t:<20} — {TASK_DESCRIPTIONS[t]}')
    readme_lines += [
        '',
        'How retrieval works:',
        '  1. The query in QUERY.txt is encoded by the embedding model.',
        '  2. All docs in the example folder are also encoded.',
        '  3. The doc with highest cosine similarity to the query is retrieved.',
        '  4. Correct = the retrieved doc is the GOLD doc.',
        '',
        f'Settings used to generate these samples:',
        f'  --corpus_size    {args.corpus_size}',
        f'  --num_examples   {args.num_examples}  (per length × depth cell)',
        f'  --cwe_occurrences {args.cwe_occurrences}',
        f'  --fwe_frequency  {args.fwe_frequency}',
        f'  --seed           {args.seed}',
    ]
    (out_root / 'README.txt').write_text('\n'.join(readme_lines), encoding='utf-8')

    for task in args.tasks:
        task_dir = out_root / task
        task_dir.mkdir(exist_ok=True)

        # Per-task README
        (task_dir / 'TASK_INFO.txt').write_text(
            f'Task: {task}\n{"="*60}\n{TASK_DESCRIPTIONS[task]}\n\n'
            + (_task_needle_guide(task)),
            encoding='utf-8')

        for context_len in args.lengths:
            for depth in args.depths:
                rng = random.Random(args.seed ^ hash((task, context_len, depth)) & 0xFFFF)

                # Pre-build distractors (same for all examples in this cell)
                distractors = []
                for i in range(args.corpus_size - 1):
                    s = args.seed + i * 9973 + context_len
                    if task.startswith('niah'):
                        distractors.append(build_niah_distractor(haystack, context_len, s))
                    elif task == 'cwe':
                        distractors.append(build_cwe_distractor(english_words, rng, context_len, s))
                    elif task == 'fwe':
                        distractors.append(build_fwe_distractor(rng, context_len, s))

                for ex_idx in range(args.num_examples):
                    s   = args.seed + ex_idx * 1009 + int(depth * 100) + context_len
                    exrng = random.Random(s)

                    needle_text, keys = None, None

                    if task.startswith('niah'):
                        gold_doc, query, gold_answers, needle_text, keys = build_niah_gold(
                            haystack, exrng, context_len, depth, task, english_words, s)
                    elif task == 'cwe':
                        gold_doc, query, gold_answers = build_cwe_gold(
                            english_words, exrng, context_len, depth, s)
                    elif task == 'fwe':
                        gold_doc, query, gold_answers = build_fwe_gold(exrng, context_len, depth, s)

                    write_example(
                        task_dir, ex_idx, task,
                        context_len, depth,
                        gold_doc, distractors,
                        query, gold_answers,
                        needle_text, keys,
                        gold_pos=0)  # will be re-randomised inside write_example

                print(f'  ✓  {task}  len={context_len}  depth={depth:.0%}  '
                      f'{args.num_examples} examples written')

    print(f'\nAll samples written to: {out_root.resolve()}')
    print('Open README.txt for orientation, then explore per-task folders.')


def _task_needle_guide(task: str) -> str:
    if not task.startswith('niah'):
        if task == 'cwe':
            return (
                'NEEDLE FORMAT\n'
                '─────────────\n'
                'No classic needle. The gold doc is a numbered word list where target\n'
                f'words each appear {args.cwe_occurrences} times. Distractors have the same format\n'
                'but no elevated frequency.\n\n'
                'WHAT TO LOOK FOR IN THE GOLD DOC\n'
                '─────────────────────────────────\n'
                'Ctrl+F the target word names in QUERY.txt inside the gold doc.\n'
                'They will appear many more times than in distractor docs.\n'
            )
        if task == 'fwe':
            return (
                'NEEDLE FORMAT\n'
                '─────────────\n'
                'No classic needle. The gold doc is a coded text ("word1....word2....") where\n'
                f'target coded-words each appear {args.fwe_frequency} times.\n\n'
                'WHAT TO LOOK FOR IN THE GOLD DOC\n'
                '─────────────────────────────────\n'
                'Ctrl+F the coded-word names from QUERY.txt inside the gold doc.\n'
                'They will appear many more times than in distractor docs.\n'
            )
        return ''

    n_pairs, val_type = NIAH_CONFIG[task]
    type_str          = NEEDLE_TYPE_STR[val_type]
    return (
        'NEEDLE FORMAT\n'
        '─────────────\n'
        f'  "The special magic {type_str} for <key> is <value>."\n\n'
        f'There are {n_pairs} needle(s) per gold document.\n'
        f'Value type: {val_type} ({type_str})\n\n'
        'WHAT TO LOOK FOR IN THE GOLD DOC\n'
        '─────────────────────────────────\n'
        'Ctrl+F "The special magic" to find the needle(s) in the gold doc.\n'
        'Distractor docs contain only Paul Graham essay text — no such sentence.\n\n'
        'HOW THE QUERY WORKS\n'
        '───────────────────\n'
        f'Query: "What is the special magic {type_str} for <key>?"\n'
        'The embedding model encodes this query and must rank the gold doc highest\n'
        'because it contains the answering needle sentence.\n'
    )


if __name__ == '__main__':
    main()