"""Benchmark submission for Entity Linking Challenge."""
from pathlib import Path

import dill as pickle
import pandas as pd
from loguru import logger
from more_itertools import chunked
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")
LINKER_PATH = Path("linker.pickle")
CER_MODEL_PATH = Path("best_yikuan8-Clinical-Longformer-noLoRA_fold4")

CONTEXT_WINDOW_WIDTH = 12
MAX_SEQ_LEN = 4096
USE_LORA = False

def load_cer_pipeline():
    label2id = {"O": 0, "B-clinical_entity": 1, "I-clinical_entity": 2}

    id2label = {v: k for k, v in label2id.items()}

    cer_tokenizer = AutoTokenizer.from_pretrained(
        CER_MODEL_PATH, model_max_length=MAX_SEQ_LEN
    )

    if USE_LORA:
        config = PeftConfig.from_pretrained(CER_MODEL_PATH)

        cer_model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=config.base_model_name_or_path,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
        )
        cer_model = PeftModel.from_pretrained(cer_model, CER_MODEL_PATH)
    else:
        cer_model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=CER_MODEL_PATH,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
        )

    cer_pipeline = pipeline(
        task="token-classification",
        model=cer_model,
        tokenizer=cer_tokenizer,
        aggregation_strategy="simple",
        device="cpu",
    )
    return cer_pipeline

def is_pos_in_span(pos, span):
    return span["start"] <= pos < span["end"]

def main():
    # columns are note_id, text
    logger.info("Reading in notes data.")
    notes = pd.read_csv(NOTES_PATH)
    logger.info(f"Found {notes.shape[0]} notes.")
    spans = []

    # Load model components
    logger.info("Loading CER pipeline.")
    cer_pipeline = load_cer_pipeline()
    cer_tokenizer = cer_pipeline.tokenizer

    logger.info("Loading Linker")
    with open(LINKER_PATH, "rb") as f:
        linker = pickle.load(f)

    # Process one note at a time...
    logger.info("Processing notes.")
    for row in notes.itertuples():
        # Tokenize the entire discharge note
        tokenized = cer_tokenizer(
            row.text,
            return_offsets_mapping=False,
            add_special_tokens=False,
            truncation=False,
        )

        global_offset = 0
        global_offset_mapping = []

        # Adjust the token offsets so that they match the original document
        for input_id in tokenized["input_ids"]:
            token = cer_tokenizer.decode(input_id)
            pos = row.text[global_offset:].find(token)
            start = global_offset + pos
            end = global_offset + pos + len(token)
            global_offset = end
            global_offset_mapping.append((start, end))

        chunk_start_idx = 0
        chunk_end_idx = 0

        # Process the document in chunks of 512 tokens chunk at a time
        for offset_chunk in chunked(global_offset_mapping, MAX_SEQ_LEN - 2):
            chunk_start_idx = chunk_end_idx
            chunk_end_idx = offset_chunk[-1][1]
            chunk_text = row.text[chunk_start_idx:chunk_end_idx]

            predicted_annotations = cer_pipeline(chunk_text)

            # combine spans that are sequential (with no space between them)
            for i in range(len(chunk_text)):
                for j in range(len(predicted_annotations)-1):
                    if is_pos_in_span(i, predicted_annotations[j]) and is_pos_in_span(i+1, predicted_annotations[j+1]):
                        predicted_annotations[j]["end"] = predicted_annotations[j+1]["end"]
                        predicted_annotations.pop(j+1)
                        break
            
            # remove invert spans
            for ann in predicted_annotations:
                if ann["start"] >= ann["end"]:
                    predicted_annotations.remove(ann)

            # ...one matched clinical entity at a time
            # Iterate through the detected entities and link them
            for entity in predicted_annotations:
                example = pd.Series(
                    {
                        # +1 to account for the [CLS] token
                        "start": entity["start"] + chunk_start_idx, # took +1 out for longformer
                        "end": entity["end"] + chunk_start_idx,
                        "text": row.text,
                    }
                )
                sctid = linker.link(example)
                if sctid:
                    spans.append(
                        {
                            "note_id": row.note_id,
                            "start": example["start"],
                            "end": example["end"],
                            "concept_id": sctid,
                        }
                    )

    logger.info(f"Generated {len(spans)} annotated spans.")
    spans_df = pd.DataFrame(spans)
    spans_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Finished.")


if __name__ == "__main__":
    main()