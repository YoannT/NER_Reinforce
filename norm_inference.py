import string
import logging
import pandas as pd

import sys
sys.path.insert(0,'/home/ytaille/deep_multilingual_normalization')

from nlstruct.utils import torch_clone
from nlstruct.utils import torch_global as tg
from deep_multilingual_normalization.preprocess import preprocess, load_quaero
from deep_multilingual_normalization.train import train_step1, train_step2, clear
from deep_multilingual_normalization.eval import compute_scores, predict

from transformers import AutoModel
from deep_multilingual_normalization.model import Classifier, FastClusteredIPSearch

from notebook_utils import *

bert_name = "camembert-base"

dataset = load_from_brat("/home/ytaille/data/tmp/ws_inputs")

docs, sentences, tokens, deltas, vocs = preprocess(
    dataset=dataset,
    max_sentence_length=120,
    bert_name=bert_name,
    vocabularies=None,
)

prep = Dataset(
    sentences=sentences,
    tokens=tokens,
    deltas=deltas,
)

batcher, encoded, ids = make_batcher(docs, sentences, tokens)

# train_batcher, vocabularies, train_mentions, train_mention_ids, group_label_mask, quaero_batcher, quaero_mentions, quaero_mention_ids = preprocess(
#     bert_name=bert_name,
#     umls_versions=["2014AB"],
#     source_full_lexicon=True,
#     source_lat=["FRE"],
#     add_quaero_splits=["train"],
#     other_full_lexicon=True,
#     other_lat=["ENG"],
#     other_additional_labels=None,
#     other_mirror_existing_labels=True,
#     sty_groups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
#     other_sabs=["CHV", "SNOMEDCT_US", "MTH", "NCI", "MSH"],
#     subs=subs,
#     apply_unidecode=True,
#     max_length=100,
# )



# classifier = Classifier(
#     n_tokens=len(vocabularies["token"]),
#     token_dim=1024 if "large" in bert_name else 768,
#     n_labels=len(vocabularies['label']),

#     ##############
#     # EMBEDDINGS #
#     ##############
#     embeddings=torch_clone(BERTS[bert_name]),

#     dropout=dropout,
#     hidden_dim=dim,
#     metric=metric,
#     metric_fc_kwargs={"cluster_label_mask": torch.as_tensor(group_label_mask.toarray()), "rescale": rescale},
#     loss='cross_entropy',
#     mask_and_shuffle=mask_and_shuffle,

#     batch_norm_affine=batch_norm_affine,
#     batch_norm_momentum=batch_norm_momentum,
# ).eval().to('cpu')
