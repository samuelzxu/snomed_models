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
import wandb
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
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

def main(
        cer_model_id, 
        use_LoRA=True, 
        random_seed=42, 
        run_suffix='', 
        max_seq_len=512, # Maximum sequence length is for (BERT-based) encoders,
        fp16=False,
        ):
    # cer_model_id = "microsoft/deberta-v3-large"  # Base model for Clinical Entity Recogniser
    # cer_model_id = "michiyasunaga/BioLinkBERT-base"  # Base model for Clinical Entity Recogniser
    kb_embedding_model_id = ("sentence-transformers/all-MiniLM-L6-v2") # base model for concept encoder

    torch.manual_seed(random_seed)
    assert torch.cuda.is_available()

    notes_df = load_notes()
    print(f"{notes_df.shape[0]} notes loaded.")

    annotations_df = load_annotations()
    print(f"{annotations_df.shape[0]} annotations loaded.")
    print(f"{annotations_df.concept_id.nunique()} unique concepts seen.")
    print(f"{annotations_df.index.nunique()} unique notes seen.")

    label2id = {"O": 0, "B-clinical_entity": 1, "I-clinical_entity": 2}

    id2label = {v: k for k, v in label2id.items()}

    cer_tokenizer = AutoTokenizer.from_pretrained(
        cer_model_id, model_max_length=max_seq_len
    )

    counts = {
        "O": 0,
        "B-clinical_entity": 0,
        "I-clinical_entity": 0,
    }

    # get data together for fivefold validation
    tokenized_df = pd.DataFrame(
        list(generate_ner_dataset(notes_df, annotations_df, cer_tokenizer, counts, max_seq_len, label2id))
    )
    print(len(tokenized_df))

    note_ids = notes_df.index.unique().to_series()

    fFold = KFold(5, shuffle=True, random_state=random_seed)

    for i, (train_ind, test_ind) in enumerate(fFold.split(note_ids)):

        train_note_ids = note_ids[train_ind]
        test_note_ids = note_ids[test_ind]
        
        train = tokenized_df[[nid in train_note_ids for nid in tokenized_df['note_id']]]
        train = Dataset.from_pandas(train)

        test = tokenized_df[[nid in test_note_ids for nid in tokenized_df['note_id']]]
        test = Dataset.from_pandas(test)

        # The data collator handles batching for us.
        data_collator = DataCollatorForTokenClassification(tokenizer=cer_tokenizer)

        os.environ["WANDB_PROJECT"] = "snoprob"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

        cer_model = AutoModelForTokenClassification.from_pretrained(
            cer_model_id, num_labels=3, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        if use_LoRA:
            lora_config = LoraConfig(
                lora_alpha=8,
                lora_dropout=0.1,
                r=8,
                bias="none",
                task_type="TOKEN_CLS",
                target_modules=["query","value","key","dense","Linear"],
            )

            cer_model = get_peft_model(cer_model, lora_config)

            cer_model.print_trainable_parameters()
        
        seqeval = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            true_labels = [
                [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = seqeval.compute(predictions=true_predictions, references=true_labels)

            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        run_name = '-'.join(cer_model_id.split('/'))+run_suffix

        wandb.init(name=run_name+f"_fold{i}")

        training_args = TrainingArguments(
            output_dir=f"~/checkpoints/{run_name}_fold{i}",
            learning_rate=2e-5,
            auto_find_batch_size=True,
            lr_scheduler_type='cosine',
            optim='adafactor',
            num_train_epochs=15,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            run_name=run_name+f"_fold{i}",
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            fp16=fp16,
            seed=random_seed,
            report_to='wandb'
        )

        decay_parameters = get_parameter_names(cer_model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in cer_model.named_parameters() if n in decay_parameters],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in cer_model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        }
        optimizer_kwargs["lr"] = training_args.learning_rate
        
        adam_bnb_optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            lr=training_args.learning_rate,
        )

        trainer = Trainer(
            model=cer_model,
            args=training_args,
            train_dataset=train,
            eval_dataset=test,
            tokenizer=cer_tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(adam_bnb_optim, None)
        )
        trainer.train()

        trainer.save_model(f"best_{run_name}_fold{i}")
        cer_tokenizer.save_pretrained(f"best_{run_name}_fold{i}")
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
