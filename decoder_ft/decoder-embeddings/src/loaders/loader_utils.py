from typing import List, Dict, Optional
import random

# ── Filename-prefix to dataset attributes ─────────────────────────────────────
# Matched left-to-right; first prefix that matches wins.
# Covers every filename in the bge_en_icl data config.
_FILENAME_PREFIX_ATTRS = [
    # ── Retrieval ──────────────────────────────────────────────────────────────
    ("msmarco",
     {"instruction": "Given a web search query, retrieve relevant passages that answer the query",
      "max_length_q": 128, "max_length_p": 512}),
    ("nq-",
     {"instruction": "Given a question, retrieve Wikipedia passages that answer the question",
      "max_length_q": 64,  "max_length_p": 384}),
    ("hotpotqa",
     {"instruction": "Given a multi-hop question, retrieve documents that can help answer the question",
      "max_length_q": 96,  "max_length_p": 384}),
    ("fiqa",
     {"instruction": "Given a financial question, retrieve user replies that best answer the question",
      "max_length_q": 64,  "max_length_p": 512}),
    ("fever",
     {"instruction": "Given a claim, retrieve documents that support or refute the claim",
      "max_length_q": 96,  "max_length_p": 512}),
    ("arguana",
     {"instruction": "Given a claim, find documents that refute the claim",
      "max_length_q": 128, "max_length_p": 512}),
    ("squad",
     {"instruction": "Given a question, retrieve Wikipedia passages that answer the question",
      "max_length_q": 64,  "max_length_p": 384}),
    ("trivial",            # TriviaQA shards are named trivial-*
     {"instruction": "Retrieve Wikipedia passages that answer the question",
      "max_length_q": 128, "max_length_p": 512}),
    ("eli5",
     {"instruction": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
      "max_length_q": 128, "max_length_p": 512}),
    ("quora",
     {"instruction": "Given a question, retrieve questions that are semantically equivalent to the given question",
      "max_length_q": 128, "max_length_p": 512}),
    ("nli",
     {"instruction": "Retrieve semantically similar text",
      "max_length_q": 128, "max_length_p": 512}),
    ("sts",
     {"instruction": "Retrieve semantically similar text",
      "max_length_q": 128, "max_length_p": 512}),
    ("scidocsrr",
     {"instruction": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
      "max_length_q": 64,  "max_length_p": 256}),
    ("stack_overflow_dup_questions",
     {"instruction": "Retrieve duplicate questions from StackOverflow forum",
      "max_length_q": 128, "max_length_p": 512}),
    # ── Science ────────────────────────────────────────────────────────────────
    ("arxiv",              # matches arXiv_abstract and arxiv_title
     {"instruction": "Given a claim or statement, retrieve documents that support the claim or statement",
      "max_length_q": 96,  "max_length_p": 512}),
    ("biorxiv",
     {"instruction": "Given a statement, retrieve related passages",
      "max_length_q": 96,  "max_length_p": 384}),
    ("medrxiv",
     {"instruction": "Given a medical claim or statement, retrieve documents that support the claim or statement",
      "max_length_q": 96,  "max_length_p": 512}),
    # ── Clustering (treated as retrieval) ─────────────────────────────────────
    ("stack_exchange_clusteringp2p",
     {"instruction": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
      "max_length_q": 128, "max_length_p": 512}),
    ("stack_exchange_clustering",
     {"instruction": "Identify the topic or theme of StackExchange posts based on the titles",
      "max_length_q": 128, "max_length_p": 256}),
    ("reddit_clusteringp2p",
     {"instruction": "Identify the topic or theme of Reddit posts based on the titles and posts",
      "max_length_q": 128, "max_length_p": 512}),
    ("reddit_clustering",
     {"instruction": "Identify the topic or theme of Reddit posts based on the titles",
      "max_length_q": 128, "max_length_p": 256}),
    # ── Classification (treated as similarity) ─────────────────────────────────
    ("amazon_counterfactual",
     {"instruction": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
      "max_length_q": 128, "max_length_p": 256}),
    ("amazon_reviews",
     {"instruction": "Classify Amazon reviews into positive or negative sentiment",
      "max_length_q": 128, "max_length_p": 256}),
    ("banking",
     {"instruction": "Given an online banking query, find the corresponding intents",
      "max_length_q": 128, "max_length_p": 256}),
    ("emotion",
     {"instruction": "Retrieve semantically similar text",
      "max_length_q": 128, "max_length_p": 256}),
    ("imdb",
     {"instruction": "Retrieve semantically similar text",
      "max_length_q": 128, "max_length_p": 512}),
    ("mtop_intent",
     {"instruction": "Classify the intent of the given utterance in task-oriented conversation",
      "max_length_q": 128, "max_length_p": 256}),
    ("toxic",
     {"instruction": "Retrieve semantically similar text",
      "max_length_q": 128, "max_length_p": 256}),
    ("tweet_sentiment",
     {"instruction": "Retrieve semantically similar text",
      "max_length_q": 128, "max_length_p": 256}),
    ("twenty_news",
     {"instruction": "Identify the topic or theme of the given news articles",
      "max_length_q": 128, "max_length_p": 512}),
]

_DEFAULT_ATTRS = {
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "max_length_q": 128,
    "max_length_p": 512,
}


def get_dataset_attrs_from_filename(filename: str) -> Dict:
    """
    Match a dataset config filename (e.g. 'msmarco_passage-00001-of-00010.jsonl.gz')
    against _FILENAME_PREFIX_ATTRS and return the matching instruction/max_length dict.
    Case-insensitive; first match wins.
    """
    fname = filename.lower()
    for prefix, attrs in _FILENAME_PREFIX_ATTRS:
        if fname.startswith(prefix.lower()):
            return attrs
    return _DEFAULT_ATTRS

def insert_lmk_tokens(
    input_ids: List[int],
    granularity: int,
    lmk_token_id: int,   # eos_token_id
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


def sample_lmk_granularity(
    granularity: int,
    granularity_set: Optional[List[int]],
) -> int:
    if granularity_set:
        return random.choice(granularity_set)
    return granularity

dataset_attributes = {
    "10": #PAQ
        {"instruction": "Given a question, retrieve passages that answer the question",
          "max_length_q": 64,
          "max_length_p": 256
        },
    "11": #WikiAnswers_pairs
        {"instruction": "Given a question, retrieve similiar questions", 
          "max_length_q": 64,
          "max_length_p": 64
        },
    "12": #S2ORC_title_abstract S2ORC_title_abstract-10M
        {"instruction": "Given a statement, retrieve related passages",
          "max_length_q": 96,
          "max_length_p": 384
        },
    "13": #S2ORC_citation_pairs
        {"instruction": "Given a statement, retrieve related statements",
          "max_length_q": 96,
          "max_length_p": 96
        },
    "14": #NQ
        {"instruction": "Given a question, retrieve Wikipedia passages that answer the question",
          "max_length_q": 64,
          "max_length_p": 384
        },
    "15": #Squad
        {"instruction": "Given a question, retrieve Wikipedia passages that answer the question",
          "max_length_q": 64,
          "max_length_p": 384
        },
    "17": #stackexchange_duplicate_questions_body_body
        {"instruction": "Given a statement, retrieve similiar statement",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "18": #stackexchange_duplicate_questions_title_title
        {"instruction": "Given a question, retrieve similiar questions",
          "max_length_q": 64,
          "max_length_p": 64
        },
    "19": #SearchQA
        {"instruction": "Given a question, retrieve passages that answer the question",
          "max_length_q": 96,
          "max_length_p": 256
        },
    "20": #specter_uniq_queries
        {"instruction": "Given a statement, retrieve related statements",
          "max_length_q": 64,
          "max_length_p": 64
        },
    # "20": #NfCorpus
    #     {"instruction": "Given a question, retrieve relevant documents that best answer the question",
    #       "max_length_q": 64,
    #       "max_length_p": 512
    #     },
    "21": #FiQA
        {"instruction": "Given a financial question, retrieve user replies that best answer the question",
          "max_length_q": 64,
          "max_length_p": 512 
        },
    "22": #S2ORC_citations_abstracts
        {"instruction": "Given a passage, retrieve related passages",
          "max_length_q": 128,
          "max_length_p": 512
        },
    # "22": #SciFact
    #     {"instruction": "Given a scientific claim, retrieve documents that support or refute the claim",
    #       "max_length_q": 128,
    #       "max_length_p": 512
    #     },
    "23":  #HotpotQA
        {"instruction": "Given a multi-hop question, retrieve documents that can help answer the question",
          "max_length_q": 96,
          "max_length_p": 384
        },
    "24": #Fever
        {"instruction": "Given a claim, retrieve documents that support or refute the claim",
          "max_length_q": 96,
          "max_length_p": 512
        },
    "25": #stackexchange_title_body
        {"instruction": "Given a claim or statement, retrieve documents that support the claim or statement",
          "max_length_q": 64,
          "max_length_p": 512
        },
    "26": #stackoverflow.com-Posts
        {"instruction": "Given a question, retrieve passages that answer the question",
          "max_length_q": 64,
          "max_length_p": 512
        },
    "28": #stackexchange_Title_Answer
        {"instruction": "Given a question, retrieve passages that answer the question",
          "max_length_q": 64,
          "max_length_p": 512
        },
    "29": #stackexchange_math
        {"instruction": "Given a math question, retrieve passages that answer the question",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "31": #wikipedia
        {"instruction": "Given a question, retrieve Wikipedia passages that answer the question",
          "max_length_q": 64,
          "max_length_p": 512
        },
    "32": #wikipedia_sections
        {"instruction": "Given a statement, retrieve related passages",
          "max_length_q": 64,
          "max_length_p": 512
        },
    "33": #Cord19
        {"instruction": "Given a query on COVID-19, retrieve documents that answer the query",
          "max_length_q": 96,
          "max_length_p": 512
        },
    "34": #Arxiv
        {"instruction": "Given a claim or statement, retrieve documents that support the claim or statement",
          "max_length_q": 96,
          "max_length_p": 512
        },
    "38": #PubMed
        {"instruction": "Given a medical claim or statement, retrieve documents that support the claim or statement",
          "max_length_q": 96,
          "max_length_p": 512
        },
    "45": #stackexchange_TitleBody_Answer
        {"instruction": "Given a question, retrieve passages that answer the question",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "50": #eli5
        {"instruction": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "51": #mr_tydi
        {"instruction": "Given a question, retrieve Wikipedia passages that answer the question",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "52": #msmarco_document, msmarco_passage
        {"instruction": "Given a web search query, retrieve relevant passages that answer the query",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "53": #nli
        {"instruction": "Retrieve semantically similar text",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "54": #quora
        {"instruction": "Given a question, retrieve questions that are semantically equivalent to the given question",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "56": #TriviaQA
        {"instruction": "Retrieve Wikipedia passages that answer the question",
          "max_length_q": 128,
          "max_length_p": 512
        },
    "99": #arguana synthetic
        {"instruction": "Given a claim, find documents that refute the claim",
          "max_length_q": 128,
          "max_length_p": 512
        },
}


# def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
#     return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


# def group_doc_ids(examples: Dict[str, List],
#                   negative_size: int,
#                   offset: int,
#                   use_first_positive: bool = False) -> List[int]:
#     pos_doc_ids: List[int] = []
#     positives: List[Dict[str, List]] = examples['positives']
#     for idx, ex_pos in enumerate(positives):
#         all_pos_doc_ids = ex_pos['doc_id']

#         if use_first_positive:
#             # keep positives that has higher score than all negatives
#             all_pos_doc_ids = [doc_id for p_idx, doc_id in enumerate(all_pos_doc_ids)
#                                if p_idx == 0 or ex_pos['score'][p_idx] >= ex_pos['score'][0]
#                                or ex_pos['score'][p_idx] > max(examples['negatives'][idx]['score'])]

#         cur_pos_doc_id = _slice_with_mod(all_pos_doc_ids, offset=offset, cnt=1)[0]
#         pos_doc_ids.append(int(cur_pos_doc_id))

#     neg_doc_ids: List[List[int]] = []
#     negatives: List[Dict[str, List]] = examples['negatives']
#     for ex_neg in negatives:
#         cur_neg_doc_ids = _slice_with_mod(ex_neg['doc_id'],
#                                           offset=offset * negative_size,
#                                           cnt=negative_size)
#         cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
#         neg_doc_ids.append(cur_neg_doc_ids)

#     assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
#     assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

#     input_doc_ids: List[int] = []
#     for pos_doc_id, neg_ids in zip(pos_doc_ids, neg_doc_ids):
#         input_doc_ids.append(pos_doc_id)
#         input_doc_ids += neg_ids

#     return input_doc_ids

# def _get_detailed_instruct(task_description: str, query: str) -> str:
#     return f'Instruct: {task_description}\nQuery: {query}'

def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    try:
      return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]
    except:
      print('ERROR', elements)
      return elements
        


def group_doc_ids(examples: Dict[str, List],
                  negative_size: int,
                  offset: int,
                  use_first_positive: bool = False) -> List[int]:
    pos_doc_ids: List[int] = []
    positives: List[Dict[str, List]] = examples['positives']
    for idx, ex_pos in enumerate(positives):
        all_pos_doc_ids = ex_pos['doc_id']

        if use_first_positive:
            # keep positives that has higher score than all negatives
            all_pos_doc_ids = [doc_id for p_idx, doc_id in enumerate(all_pos_doc_ids)
                               if p_idx == 0 or ex_pos['score'][p_idx] >= ex_pos['score'][0]
                               or ex_pos['score'][p_idx] > max(examples['negatives'][idx]['score'])]

        cur_pos_doc_id = _slice_with_mod(all_pos_doc_ids, offset=offset, cnt=1)[0]
        # if isinstance(cur_pos_doc_id, str):
        #     cur_pos_doc_id = random.randint(10**9, 10**12)
        # pos_doc_ids.append(int(cur_pos_doc_id))
        pos_doc_ids.append(cur_pos_doc_id)

    neg_doc_ids: List[List[int]] = []
    negatives: List[Dict[str, List]] = examples['negatives']
    for ex_neg in negatives:
        cur_neg_doc_ids = _slice_with_mod(ex_neg['doc_id'],
                                          offset=offset * negative_size,
                                          cnt=negative_size)
        # cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
        neg_doc_ids.append(cur_neg_doc_ids)

    assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
    assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

    input_doc_ids: List[int] = []
    for pos_doc_id, neg_ids in zip(pos_doc_ids, neg_doc_ids):
        input_doc_ids.append(pos_doc_id)
        input_doc_ids += neg_ids

    return input_doc_ids

def _get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'