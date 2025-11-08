"""
build_qa_dataset.py - Generate a simple algorithmic Q&A dataset.

This script creates a dataset for tasks like identifying a word's position,
counting words, or finding a subsequent word in a randomly generated sentence.
The format is inspired by the Sudoku dataset generator.
"""

import json
import numpy as np
import os
import random
from typing import List, Tuple, Dict
from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata


cli = ArgParser()


class QADatasetConfig(BaseModel):
    output_dir: str = "data/simple_qa"
    max_seq_len: int = 48  # Max length for padded sequences (Q + Sentence + A)
    num_train_puzzles: int = 5000
    num_test_puzzles: int = 1000
    seed: int = 42
    vocab_size: int = 100  # Number of unique words in the vocabulary
    min_sentence_words: int = 5
    max_sentence_words: int = 16
    num_qa_templates: int = 3  # Number of different Q&A patterns


# --- Vocabulary and Tokenization ---

def build_vocabulary(vocab_size: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Builds a vocabulary of random words and special tokens."""
    special_tokens = ["[PAD]", "[EOS]", "[SEP]", "[UNK]"]
    words = [f"word{i}" for i in range(vocab_size - len(special_tokens))]
    
    vocab = special_tokens + words
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    
    return token_to_id, id_to_token


# --- Q&A Template Generation ---

def generate_random_sentence(words: List[str], min_len: int, max_len: int) -> List[str]:
    """Generates a sentence with a random sequence of words."""
    sentence_len = random.randint(min_len, max_len)
    return random.sample(words, sentence_len)


def qa_template_find_nth_word(sentence: List[str]) -> Tuple[str, str]:
    """Q&A for finding the Nth word."""
    n = random.randint(1, len(sentence))
    question = f"what is the {n}th word in the sentence ?"
    answer = sentence[n-1]
    return question, answer


def qa_template_count_words(sentence: List[str]) -> Tuple[str, str]:
    """Q&A for counting the number of words."""
    question = "how many words are in the sentence ?"
    answer = str(len(sentence))
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
]


# --- Data Encoding and Padding ---

def encode_and_pad(
    question: str, sentence: List[str], answer: str, 
    token_to_id: Dict[str, int], max_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes text into token IDs, pads, and creates labels."""
    
    # Tokenize
    q_tokens = question.split()
    s_tokens = sentence
    a_tokens = answer.split()
    
    # Combine input sequence: Question [SEP] Sentence
    input_tokens = q_tokens + ["[SEP]"] + s_tokens
    input_ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in input_tokens]
    
    # Create labels (predicting the answer)
    label_ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in a_tokens]
    
    # Pad inputs
    padded_input = np.full(max_len, token_to_id["[PAD]"], dtype=np.int32)
    input_len = min(len(input_ids) + 1, max_len) # +1 for EOS
    padded_input[:input_len-1] = input_ids[:input_len-1]
    padded_input[input_len-1] = token_to_id["[EOS]"]

    # Pad labels (ignore loss on non-answer parts)
    padded_labels = np.full(max_len, 0, dtype=np.int32) # 0 is ignore_label_id
    label_len = min(len(label_ids), max_len - input_len)
    padded_labels[input_len : input_len + label_len] = label_ids[:label_len]

    return padded_input, padded_labels


# --- Main Dataset Generation ---

def generate_dataset(config: QADatasetConfig, split: str, token_to_id: Dict[str, int]):
    """Generate the Q&A dataset."""
    np.random.seed(config.seed + (0 if split == "train" else 1000))
    random.seed(config.seed + (0 if split == "train" else 1000))
    
    num_puzzles = config.num_train_puzzles if split == "train" else config.num_test_puzzles
    words = [key for key in token_to_id.keys() if key.startswith("word")]

    all_inputs = []
    all_labels = []
    puzzle_identifiers = []
    puzzle_indices = [0]
    group_indices = [0]
    
    example_count = 0
    
    for i in range(num_puzzles):
        if i % 500 == 0:
            print(f"  Generating {split} example {i}/{num_puzzles}")
            
        # 1. Generate a random sentence
        sentence = generate_random_sentence(words, config.min_sentence_words, config.max_sentence_words)
        
        # 2. Pick a random Q&A template
        template = random.choice(QA_TEMPLATES)
        question, answer = template(sentence)
        
        # 3. Encode and pad
        inp, lbl = encode_and_pad(question, sentence, answer, token_to_id, config.max_seq_len)
        all_inputs.append(inp)
        all_labels.append(lbl)
        example_count += 1

        puzzle_indices.append(example_count)
        puzzle_identifiers.append(0)  # All Q&A puzzles have ID 0
        group_indices.append(i + 1)

    # Save dataset
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
        
    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_seq_len,
        vocab_size=len(token_to_id),
        pad_id=token_to_id["[PAD]"],
        ignore_label_id=0, # Using 0 for padding in labels
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_puzzles=num_puzzles,
        mean_puzzle_examples=1,
        total_groups=num_puzzles,
        sets=["all"]
    )
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
        
    print(f"\nGenerated {split} split:")
    print(f"  Examples: {num_puzzles}")
    print(f"  Input shape: {results_np['inputs'].shape}")


@cli.command(singleton=True)
def main(config: QADatasetConfig):
    """Generate Simple Algorithmic Q&A Dataset."""
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    print("Generating Simple Q&A dataset...")
    print(f"Configuration:")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Vocabulary size: {config.vocab_size}")

    # Build vocabulary
    token_to_id, id_to_token = build_vocabulary(config.vocab_size)
    
    # Generate datasets
    generate_dataset(config, "train", token_to_id)
    generate_dataset(config, "test", token_to_id)
    
    # Save vocabulary and identifiers
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token}, f, indent=2)
        
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "qa"], f)

    print("\n" + "="*50)
    print("Dataset generated successfully!")


if __name__ == "__main__":
    cli()
