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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DebertaV2ForTokenClassification,
    Trainer,
    TrainingArguments,
    AutoModel,
    pipeline,
)

from snomed_graph import *
from utils import load_notes, load_annotations
from constants import label2id, id2label

class Linker:
    def __init__(self, encoder, context_window_width=0):
        self.encoder = encoder
        self.entity_index = KeyedVectors(self.encoder[1].word_embedding_dimension)
        self.context_index = dict()
        self.history = dict()
        self.context_window_width = context_window_width

    def add_context(self, row):
        window_start = max(0, row.start - self.context_window_width)
        window_end = min(row.end + self.context_window_width, len(row.text))
        return row.text[window_start:window_end]

    def add_entity(self, row):
        return row.text[row.start : row.end]

    def fit(self, df=None, snomed_concepts=None):
        # Create a map from the entities to the concepts and contexts in which they appear
        if df is not None:
            for row in df.itertuples():
                entity = self.add_entity(row)
                context = self.add_context(row)
                map_ = self.history.get(entity, dict())
                contexts = map_.get(row.concept_id, list())
                contexts.append(context)
                map_[row.concept_id] = contexts
                self.history[entity] = map_

        # Add SNOMED CT codes for lookup
        if snomed_concepts is not None:
            for c in snomed_concepts:
                for syn in c.synonyms:
                    map_ = self.history.get(syn, dict())
                    contexts = map_.get(c.sctid, list())
                    contexts.append(syn)
                    map_[c.sctid] = contexts
                    self.history[syn] = map_

        # Create indexes to help disambiguate entities by their contexts
        for entity, map_ in tqdm(self.history.items()):
            keys = [
                (concept_id, occurance)
                for concept_id, contexts in map_.items()
                for occurance, context in enumerate(contexts)
            ]
            contexts = [context for contexts in map_.values() for context in contexts]
            vectors = self.encoder.encode(contexts)
            index = KeyedVectors(self.encoder[1].word_embedding_dimension)
            index.add_vectors(keys, vectors)
            self.context_index[entity] = index

        # Now create the top-level entity index
        keys = list(self.history.keys())
        vectors = self.encoder.encode(keys)
        self.entity_index.add_vectors(keys, vectors)

    def link(self, row):
        entity = self.add_entity(row)
        context = self.add_context(row)
        vec = self.encoder.encode(entity)
        nearest_entity = self.entity_index.most_similar(vec, topn=1)[0][0]
        index = self.context_index.get(nearest_entity, None)

        if index:
            vec = self.encoder.encode(context)
            key, score = index.most_similar(vec, topn=1)[0]
            sctid, _ = key
            return sctid
        else:
            return None

def main(encoder_model_id, random_seed=42, scope="min"):

    encoder_model_id = (encoder_model_id)

    torch.manual_seed(random_seed)
    assert torch.cuda.is_available()

    notes_df = load_notes()
    annotations_df = load_annotations()

    training_notes_df, test_notes_df = train_test_split(
        notes_df, test_size=32, random_state=random_seed
    )
    training_annotations_df = annotations_df.loc[training_notes_df.index]
    test_annotations_df = annotations_df.loc[test_notes_df.index]

    print(
        f"There are {training_annotations_df.shape[0]} total annotations in the training set."
    )
    print(f"There are {test_annotations_df.shape[0]} total annotations in the test set.")
    print(
        f"There are {training_annotations_df.concept_id.nunique()} distinct concepts in the training set."
    )
    print(
        f"There are {test_annotations_df.concept_id.nunique()} distinct concepts in the test set."
    )
    print(f"There are {training_notes_df.shape[0]} notes in the training set.")
    print(f"There are {test_notes_df.shape[0]} notes in the test set.")

    kb_model = SentenceTransformer(encoder_model_id)
    SG = SnomedGraph.from_serialized("./full_concept_graph.gml")

    # If we want to simply use concepts for which we have a training example, it's this:
    concepts_in_scope = [
        SG.get_concept_details(a) for a in annotations_df.concept_id.unique()
    ]

    print(f"{len(concepts_in_scope)} concepts have been selected.")

    kb_sft_examples = [
        InputExample(texts=[syn1, syn2], label=1)
        for concept in tqdm(concepts_in_scope)
        for syn1, syn2 in combinations(concept.synonyms, 2)
    ]

    kb_sft_dataloader = DataLoader(kb_sft_examples, shuffle=True, batch_size=96)

    kb_sft_loss = losses.ContrastiveLoss(kb_model)

    # THERE IS A MUCH BETTER WAY OF CONSTRUCTING THIS DATASET!

    kb_model.fit(
        train_objectives=[(kb_sft_dataloader, kb_sft_loss)],
        epochs=2,
        warmup_steps=100,
        checkpoint_path="temp/ke_encoder",
    )

    kb_model.save("kb_model")