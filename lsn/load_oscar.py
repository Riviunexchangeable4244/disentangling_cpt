#!/usr/bin/env python3
"""
Preprocess multilingual OSCAR data for model training.

This script performs:
1. Downloads the OSCAR dataset for the specified languages:
   https://huggingface.co/datasets/oscar-corpus
2. Tokenizes text using a Hugging Face tokenizer
3. Aggregates up to TARGET_TOKENS_PER_LANGUAGE
4. Saves tokenized data as PyTorch tensors
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import argparse

NUM_PROC_BASE = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
TARGET_TOKENS_PER_LANGUAGE = 100_000_000

def tokenize_function(examples, tokenizer):
    texts = []
    for example_texts in examples["text"]:
        for text in example_texts:
            texts.append(text["text"])
    if not texts:
        return {"input_ids": []}

    output = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )

    return {"input_ids": output.input_ids}


def build_and_save(
    lang, model_id, tokenizer_name, output_dir, num_proc_map=NUM_PROC_BASE
):
    print(f"Starting data processing for language: {lang}")

    train_filename_base = f"oscar.{lang}.train.{model_id.replace('/', '_')}"
    train_output_path = os.path.join(output_dir, train_filename_base)

    try:
        ds = load_dataset(
            "oscar-corpus/mOSCAR",
            lang,
            streaming=False,
        )['train']
        if len(ds) == 0:
            print(f"Warning: Dataset for {lang} is empty. Skipping.")
            return
    except Exception as e:
        print(f"Error loading dataset for {lang}: {e}")
        raise
    
    print(f"Dataset length is {len(ds)}")

    LIMIT = 2_000_000
    if len(ds) > LIMIT:
        ds = ds.select(range(LIMIT))

    print(f"Dataset length is {len(ds)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        raise

    tokenization_func_with_tokenizer = partial(tokenize_function, tokenizer=tokenizer)
    tokenized_ds = ds.map(
        tokenization_func_with_tokenizer,
        batched=True,
        num_proc=num_proc_map,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {lang}",
    )

    all_document_token_lists = []
    for processed_example in tqdm(
        tokenized_ds, desc=f"Collecting token lists for {lang}"
    ):
        token_list_for_one_doc = processed_example["input_ids"]
        if isinstance(token_list_for_one_doc, list):
            all_document_token_lists.append(token_list_for_one_doc)

    if not all_document_token_lists:
        print(f"Warning: No token sequences found for {lang}. Skipping.")
        return

    final_token_ids = []
    collected_tokens_count = 0
    for doc_tokens_list in tqdm(
        all_document_token_lists, desc=f"Aggregating tokens for {lang}"
    ):
        if not doc_tokens_list:
            continue

        current_doc_token_count = len(doc_tokens_list)
        if (
            collected_tokens_count + current_doc_token_count
            <= TARGET_TOKENS_PER_LANGUAGE
        ):
            final_token_ids.extend(doc_tokens_list)
            collected_tokens_count += current_doc_token_count
        else:
            remaining_needed = TARGET_TOKENS_PER_LANGUAGE - collected_tokens_count
            final_token_ids.extend(doc_tokens_list[:remaining_needed])
            collected_tokens_count += remaining_needed
            break

        if collected_tokens_count >= TARGET_TOKENS_PER_LANGUAGE:
            break

    del all_document_token_lists
    del tokenized_ds
    del ds

    if collected_tokens_count == 0:
        print(f"Warning: Zero tokens collected for {lang}. Skipping save.")
        return

    if collected_tokens_count < TARGET_TOKENS_PER_LANGUAGE:
        print(
            f"Warning: Language {lang} has only {collected_tokens_count:,} tokens, "
            f"which is less than the target of {TARGET_TOKENS_PER_LANGUAGE:,}."
        )

    full_tensor = torch.tensor(final_token_ids, dtype=torch.long)
    del final_token_ids

    os.makedirs(output_dir, exist_ok=True)
    torch.save(full_tensor, train_output_path)
    print(f"Saved {full_tensor.numel():,} tokens for {lang}.")
    del full_tensor


def run_job(args):
    lang, model_id, tokenizer_name, output_dir, num_proc_map = args
    print(f"Processing language: {lang} (PID: {os.getpid()})")
    try:
        build_and_save(lang, model_id, tokenizer_name, output_dir, num_proc_map)
        return lang, True, None
    except Exception as e:
        import traceback

        traceback.print_exc()
        return lang, False, str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OSCAR data for multiple languages."
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,zh,fr",
        help="Comma-separated list of languages to process, e.g., 'en,zh,fr'",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier (used for file naming).",
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Tokenizer name or path."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train-data",
        help="Where to store tokenized tensors.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=6, help="Max concurrent processes."
    )
    args = parser.parse_args()

    args.languages = [
        lang.strip() for lang in args.languages.split(",") if lang.strip()
    ]

    MAX_CONCURRENT_LANGUAGES = args.max_concurrent
    NUM_MAP_PROC_PER_LANG = max(
        1,
        (
            NUM_PROC_BASE // MAX_CONCURRENT_LANGUAGES
            if MAX_CONCURRENT_LANGUAGES > 0
            else NUM_PROC_BASE
        ),
    )

    print(f"Starting batch processing for {len(args.languages)} languages.")

    job_args_list = [
        (lang, args.model_id, args.tokenizer, args.output_dir, NUM_MAP_PROC_PER_LANG)
        for lang in args.languages
    ]

    successful_langs = []
    failed_langs_with_errors = {}

    with multiprocessing.Pool(processes=MAX_CONCURRENT_LANGUAGES) as pool:
        results_iterable = pool.imap_unordered(run_job, job_args_list)
        for result in tqdm(
            results_iterable,
            total=len(args.languages),
            desc="Overall Language Progress",
        ):
            lang_processed, success, error_msg = result
            if success:
                successful_langs.append(lang_processed)
            else:
                failed_langs_with_errors[lang_processed] = error_msg

    print("Batch processing finished.")
    print(f"Successfully processed: {', '.join(sorted(successful_langs))}")
    if failed_langs_with_errors:
        print(
            f"Failed to process: {', '.join(sorted(failed_langs_with_errors.keys()))}"
        )
        for lang_failed, err in failed_langs_with_errors.items():
            print(f"  - {lang_failed}: {err}")
