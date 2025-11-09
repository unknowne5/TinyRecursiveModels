"""
build_qa_variants_dataset.py - Generate variants of the structured Q&A dataset.

This script creates datasets with a specified number of words and maximum sentence length.
"""

import json
import numpy as np
import os
import random
from typing import List, Tuple, Dict
from argdantic import ArgParser
from pydantic import BaseModel, Field

from common import PuzzleDatasetMetadata


cli = ArgParser()


class QADatasetConfig(BaseModel):
    output_dir: str = Field(..., description="Directory to save the dataset.")
    num_words: int = Field(..., description="Number of words to use for generating sentences.")
    max_sentence_words: int = Field(..., description="Maximum number of words in a sentence.")
    total_vocab_words: int = Field(50, description="Total number of words in the vocabulary.")
    max_seq_len: int = 48
    num_train_puzzles: int = 8000
    num_test_puzzles: int = 1500
    seed: int = 42
    min_sentence_words: int = 5

# --- Vocabulary and Tokenization ---

def get_question_words():
    """Returns a list of all words used in question templates."""
    return sorted(list(set("what is the word in the sentence how many words are after first".split())))

def build_vocabulary(total_vocab_words: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Builds a vocabulary with single-digit numbers and a specified number of words."""
    special_tokens = ["[PAD]", "[EOS]", "[SEP]", "[UNK]"]
    number_tokens = [str(i) for i in range(10)]  # 0-9
    question_tokens = get_question_words()
    words = [f"word{i}" for i in range(1, total_vocab_words + 1)]

    vocab = special_tokens + number_tokens + question_tokens + words
    
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}

    return token_to_id, id_to_token


# --- Q&A Template Generation ---

def generate_random_sentence(words: List[str], min_len: int, max_len: int) -> List[str]:
    """Generates a sentence with a random sequence of words."""
    max_len = min(max_len, len(words))
    min_len = min(min_len, max_len)
    sentence_len = random.randint(min_len, max_len)
    return random.sample(words, sentence_len)


def qa_template_find_nth_word(sentence: List[str]) -> Tuple[str, str]:
    """Q&A for finding the Nth word."""
    n = random.randint(1, len(sentence))
    question = f"what is the {n} word in the sentence ?"
    answer = sentence[n-1]
    return question, answer


def qa_template_count_words(sentence: List[str]) -> Tuple[str, str]:
    """Q&A for counting the number of words."""
    question = "how many words are in the sentence ?"
    answer = str(len(sentence))
    return question, answer


def qa_template_find_first_n_words(sentence: List[str]) -> Tuple[str, str]:
    """Q&A for finding the first N words."""
    if len(sentence) < 2:
        return qa_template_find_nth_word(sentence) # Fallback
    n = random.randint(2, min(4, len(sentence))) # Ask for 2, 3, or 4 words
    question = f"what are the first {n} words in the sentence ?"
    answer = " ".join(sentence[:n])
    return question, answer

def qa_template_find_next_word(sentence: List[str]) -> Tuple[str, str]:
    """Q&A for finding the word that follows another."""
    if len(sentence) < 2:
        return qa_template_find_nth_word(sentence) # Fallback
    
    idx = random.randint(0, len(sentence) - 2)
    target_word = sentence[idx]
    question = f"what is the word after {target_word} in the sentence ?"
    answer = sentence[idx+1]
    return question, answer


QA_TEMPLATES = [
    qa_template_find_nth_word,
    qa_template_count_words,
    qa_template_find_next_word,
    qa_template_find_first_n_words,
]


# --- Data Encoding and Padding ---

def custom_tokenizer(text: str) -> List[str]:
    """Splits text into words and handles numbers as individual digits."""
    tokens = []
    for word in text.split():
        if word.isdigit():
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens

def encode_and_pad(
    question: str, sentence: List[str], answer: str, 
    token_to_id: Dict[str, int], max_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes text into token IDs, pads, and creates labels."""
    q_tokens = custom_tokenizer(question)
    s_tokens = sentence
    a_tokens = custom_tokenizer(answer)
    
    input_tokens = q_tokens + ["[SEP]"] + s_tokens
    input_ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in input_tokens]
    label_ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in a_tokens]
    
    padded_input = np.full(max_len, token_to_id["[PAD]"], dtype=np.int32)
    input_len = min(len(input_ids) + 1, max_len)
    padded_input[:input_len-1] = input_ids[:input_len-1]
    padded_input[input_len-1] = token_to_id["[EOS]"]

    padded_labels = np.full(max_len, token_to_id["[PAD]"], dtype=np.int32)
    label_len = min(len(label_ids), max_len)
    padded_labels[:label_len] = label_ids[:label_len]

    return padded_input, padded_labels


# --- Main Dataset Generation ---

def generate_dataset(config: QADatasetConfig, split: str, token_to_id: Dict[str, int]):
    np.random.seed(config.seed + (0 if split == "train" else 1000))
    random.seed(config.seed + (0 if split == "train" else 1000))
    
    num_puzzles = config.num_train_puzzles if split == "train" else config.num_test_puzzles
    all_words = sorted([key for key in token_to_id.keys() if key.startswith("word") and key[4:].isdigit()], key=lambda x: int(x[4:]))
    words_to_use = all_words[:config.num_words]


    all_inputs, all_labels = [], []
    puzzle_identifiers, puzzle_indices, group_indices = [], [0], [0]
    debug_samples = []
    
    for i in range(num_puzzles):
        if i % 1000 == 0:
            print(f"  Generating {split} example {i}/{num_puzzles}")
            
        sentence = generate_random_sentence(words_to_use, config.min_sentence_words, config.max_sentence_words)
        template = random.choice(QA_TEMPLATES)
        question, answer = template(sentence)
        
        inp, lbl = encode_and_pad(question, sentence, answer, token_to_id, config.max_seq_len)

        if i < 100:
            debug_samples.append({
                "id": i, "sentence": " ".join(sentence), "question": question, "answer": answer,
                "encoded_input": inp, "encoded_labels": lbl,
            })
            
        all_inputs.append(inp)
        all_labels.append(lbl)
        puzzle_indices.append(len(all_inputs))
        puzzle_identifiers.append(0)
        group_indices.append(i + 1)

    save_dir = os.path.join(config.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    
    results_np = {
        "inputs": np.array(all_inputs, dtype=np.int32),
        "labels": np.array(all_labels, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32),
    }
    
    for key, value in results_np.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), value)

    with open(os.path.join(save_dir, "debug_samples.txt"), "w") as f:
        for sample in debug_samples:
            f.write(f"--- Example {sample['id']} ---\n"
                    f"Sentence: {sample['sentence']}\n"
                    f"Question: {sample['question']}\n"
                    f"Answer:   {sample['answer']}\n"
                    f"Input:    {sample['encoded_input']}\n"
                    f"Labels:   {sample['encoded_labels']}\n\n")
        
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_seq_len, vocab_size=len(token_to_id), pad_id=token_to_id["[PAD]"],
        ignore_label_id=token_to_id["[PAD]"], blank_identifier_id=0, num_puzzle_identifiers=1,
        total_puzzles=num_puzzles, mean_puzzle_examples=1, total_groups=num_puzzles, sets=["all"]
    )
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
        
    print(f"\nGenerated {split} split: {num_puzzles} examples")


@cli.command(singleton=True)
def main(config: QADatasetConfig):
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    print(f"Generating dataset variant in {config.output_dir}...")
    
    token_to_id, id_to_token = build_vocabulary(config.total_vocab_words)
    
    generate_dataset(config, "train", token_to_id)
    generate_dataset(config, "test", token_to_id)
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token}, f, indent=2)
        
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "qa"], f)

    print("\n" + "="*50)
    print("Dataset generated successfully!")


if __name__ == "__main__":
    cli()
