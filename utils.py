from itertools import combinations

import dill as pickle
import evaluate
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datasets import Dataset
from gensim.models.keyedvectors import KeyedVectors
from ipymarkup import show_span_line_markup
from more_itertools import chunked
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DebertaForTokenClassification,
    DebertaV2ForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    AutoModel,
    pipeline,
)
import fire
import os

from snomed_graph import *

# Step through the annotation spans for a given note.  When they're exhausted,
# return (1000000, 1000000).  This will avoid a StopIteration exception.

def get_annotation_boundaries(note_id, annotations_df):
    for row in annotations_df.loc[note_id].itertuples():
        yield row.start, row.end, row.concept_id
    yield 1000000, 1000000, None

def generate_ner_dataset(notes_df, annotations_df, cer_tokenizer, counts, max_seq_len, label2id):
    for row in notes_df.itertuples():
        tokenized = cer_tokenizer(
            row.text,
            return_offsets_mapping=False,  # Avoid misalignments due to destructive tokenization
            return_token_type_ids=False,  # We're going to construct these below
            return_attention_mask=False,  # We'll construct this by hand
            add_special_tokens=False,  # We'll add these by hand
            truncation=False,  # We'll chunk the notes ourselves
        )

        # Prime the annotation generator and fetch the token <-> word_id map
        annotation_boundaries = get_annotation_boundaries(row.Index, annotations_df)
        ann_start, ann_end, concept_id = next(annotation_boundaries)
        word_ids = tokenized.word_ids()

        # The offsets_mapping returned by the tokenizer will be misaligned vs the original text.
        # This is due to the fact that the tokenization scheme is destructive, for example it
        # drops spaces which cannot be recovered when decoding the inputs.
        # In the following code snippet we create an offset mapping which is aligned with the
        # original text; hence we can accurately locate the annotations and match them to the
        # tokens.
        global_offset = 0
        global_offset_mapping = []

        for input_id in tokenized["input_ids"]:
            token = cer_tokenizer.decode(input_id)
            pos = row.text[global_offset:].find(token)
            start = global_offset + pos
            end = global_offset + pos + len(token)
            global_offset = end
            global_offset_mapping.append((start, end))

        # Note the max_seq_len - 2.
        # This is because we will have to add [CLS] and [SEP] tokens once we're done.
        it = zip(
            chunked(tokenized["input_ids"], max_seq_len - 2),
            chunked(global_offset_mapping, max_seq_len - 2),
            chunked(word_ids, max_seq_len - 2),
        )

        # Since we are chunking the discharge notes, we need to maintain the start and
        # end character index for each chunk so that we can align the annotations for
        # chunks > 1
        chunk_start_idx = 0
        chunk_end_idx = 0

        for chunk_id, chunk in enumerate(it):
            input_id_chunk, offset_mapping_chunk, word_id_chunk = chunk
            token_type_chunk = list()
            concept_id_chunk = list()
            prev_word_id = -1
            concept_word_number = 0
            chunk_start_idx = chunk_end_idx
            chunk_end_idx = offset_mapping_chunk[-1][1]

            for offsets, word_id in zip(offset_mapping_chunk, word_id_chunk):
                token_start, token_end = offsets

                # Check whether we need to fetch the next annotation
                if token_start >= ann_end:
                    ann_start, ann_end, concept_id = next(annotation_boundaries)
                    concept_word_number = 0

                # Check whether the token's position overlaps with the next annotation
                if token_start < ann_end and token_end > ann_start:
                    if prev_word_id != word_id:
                        concept_word_number += 1

                    # If so, annotate based on the word number in the concept
                    if concept_word_number == 1:
                        token_type_chunk.append(label2id["B-clinical_entity"])
                        counts["B-clinical_entity"] += 1
                    else:
                        token_type_chunk.append(label2id["I-clinical_entity"])
                        counts["I-clinical_entity"] += 1

                    # Add the SCTID (we'll use this later to train the Linker)
                    concept_id_chunk.append(concept_id)

                # Not part of an annotation
                else:
                    token_type_chunk.append(label2id["O"])
                    counts["O"] += 1
                    concept_id_chunk.append(None)

                prev_word_id = word_id

            # Manually adding the [CLS] and [SEP] tokens.
            token_type_chunk = [-100] + token_type_chunk + [-100]
            input_id_chunk = (
                [cer_tokenizer.cls_token_id]
                + input_id_chunk
                + [cer_tokenizer.sep_token_id]
            )
            attention_mask_chunk = [1] * len(input_id_chunk)
            offset_mapping_chunk = (
                [(None, None)] + offset_mapping_chunk + [(None, None)]
            )
            concept_id_chunk = [None] + concept_id_chunk + [None]

            yield {
                # These are the fields we need
                "note_id": row.Index,
                "input_ids": input_id_chunk,
                "attention_mask": attention_mask_chunk,
                "labels": token_type_chunk,
                # These fields are helpful for debugging
                "chunk_id": chunk_id,
                "chunk_span": (chunk_start_idx, chunk_end_idx),
                "offset_mapping": offset_mapping_chunk,
                "text": row.text[chunk_start_idx:chunk_end_idx],
                "concept_ids": concept_id_chunk,
            }

def load_notes():
    return pd.read_csv("data/training_notes.csv").set_index("note_id")

def load_annotations():
    return pd.read_csv("data/train_annotations.csv").set_index("note_id")