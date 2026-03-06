"""
build_vision_qa_dataset.py - Generate a simple visual Q&A dataset.

This script creates a dataset of simple images (e.g., colored squares on a black background)
and corresponding text answers (e.g., counting the squares of a specific color),
so the model can take images as inputs and text tokens as labels.
"""

import json
import numpy as np
import os
import random
from typing import List, Tuple, Dict
from argdantic import ArgParser
from pydantic import BaseModel

from PIL import Image
from common import PuzzleDatasetMetadata

cli = ArgParser()

class VisionDatasetConfig(BaseModel):
    output_dir: str = "data/vision_qa"
    max_seq_len: int = 16
    image_size: int = 32
    patch_size: int = 4
    num_train_puzzles: int = 8000
    num_test_puzzles: int = 1500
    seed: int = 42

# --- Vocabulary and Tokenization ---

def build_vocabulary() -> Tuple[Dict[str, int], Dict[int, str]]:
    special_tokens = ["[PAD]", "[EOS]", "[SEP]", "[UNK]"]
    number_tokens = [str(i) for i in range(10)]  # 0-9
    color_tokens = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    words = ["there", "are", "squares", "is", "one", "square", "zero", "how", "many", "?", "what", "color"]
    
    vocab = special_tokens + number_tokens + color_tokens + words
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    return token_to_id, id_to_token

def custom_tokenizer(text: str) -> List[str]:
    tokens = []
    for word in text.split():
        if word.isdigit():
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens

def encode_text(text: str, token_to_id: Dict[str, int], max_len: int, add_eos: bool = False) -> np.ndarray:
    tokens = custom_tokenizer(text)
    token_ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in tokens]
    
    padded = np.full(max_len, token_to_id["[PAD]"], dtype=np.int32)
    length = min(len(token_ids), max_len - (1 if add_eos else 0))
    padded[:length] = token_ids[:length]
    if add_eos:
        padded[length] = token_to_id["[EOS]"]
    
    return padded

# --- Image Generation ---

COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0)
}

def generate_image_and_qa(image_size: int) -> Tuple[np.ndarray, str, str]:
    # Create empty black image: C x H x W
    image = np.zeros((3, image_size, image_size), dtype=np.float32)
    
    color_names = list(COLORS.keys())
    target_color = random.choice(color_names)
    target_rgb = COLORS[target_color]
    
    # Randomly place 1 to 9 squares
    num_squares = random.randint(1, 9)
    actual_target_count = 0
    
    square_size = max(2, image_size // 8)
    
    for _ in range(num_squares):
        c_name = random.choice(color_names)
        c_rgb = COLORS[c_name]
        
        if c_name == target_color:
            actual_target_count += 1
            
        x = random.randint(0, image_size - square_size - 1)
        y = random.randint(0, image_size - square_size - 1)
        
        # Channel, Y, X
        image[0, y:y+square_size, x:x+square_size] = c_rgb[0]
        image[1, y:y+square_size, x:x+square_size] = c_rgb[1]
        image[2, y:y+square_size, x:x+square_size] = c_rgb[2]
        
    # Generate question/answer
    question = f"how many {target_color} squares are there ?"
    
    if actual_target_count == 1:
        answer = f"there is 1 {target_color} square"
    else:
        answer = f"there are {actual_target_count} {target_color} squares"
        
    return image, question, answer

# --- Main Dataset Generation ---

def generate_dataset(config: VisionDatasetConfig, split: str, token_to_id: Dict[str, int]):
    np.random.seed(config.seed + (0 if split == "train" else 1000))
    random.seed(config.seed + (0 if split == "train" else 1000))
    
    num_puzzles = config.num_train_puzzles if split == "train" else config.num_test_puzzles
    
    save_dir = os.path.join(config.output_dir, split)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    all_inputs, all_text_inputs, all_labels = [], [], []
    puzzle_identifiers, puzzle_indices, group_indices = [], [0], [0]
    debug_samples = []
    
    seq_len_patches = (config.image_size // config.patch_size) ** 2
    # Total seq len is text tokens + image patches
    total_seq_len = 16 + seq_len_patches
    
    for i in range(num_puzzles):
        if i % 1000 == 0:
            print(f"  Generating {split} example {i}/{num_puzzles}")
            
        image, question, answer = generate_image_and_qa(config.image_size)
        
        # Max length of 16 is enough for the question tokens
        txt_inp = encode_text(question, token_to_id, max_len=16, add_eos=False)
        # Pad the labels to the full sequence length (80)
        lbl = encode_text(answer, token_to_id, max_len=total_seq_len, add_eos=True)
        
        if i < 10:
            debug_samples.append({
                "id": i,
                "question": question,
                "answer": answer,
                "encoded_text_inputs": txt_inp.tolist(),
                "encoded_labels": lbl.tolist(),
            })
            
        # Convert float image to uint8 for saving
        # image is C x H x W in [0, 1]
        img_uint8 = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_filename = f"image_{i}.png"
        img_pil.save(os.path.join(images_dir, img_filename))
        
        # Save just the index in the numpy array instead of the full image
        all_inputs.append(i)
        all_text_inputs.append(txt_inp)
        all_labels.append(lbl)
        puzzle_indices.append(len(all_inputs))
        puzzle_identifiers.append(0)
        group_indices.append(i + 1)

    # the sequence length for vision input isn't predefined in terms of tokens, 
    # but the model patch embedding will convert it to patches.
    # We still keep seq_len for labels.
    
    results_np = {
        "inputs": np.array(all_inputs, dtype=np.int32),  # (N,) containing indices
        "text_inputs": np.array(all_text_inputs, dtype=np.int32), # (N, text_seq_len)
        "labels": np.array(all_labels, dtype=np.int32),    # (N, seq_len)
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32),
    }
    
    for key, value in results_np.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), value)

    with open(os.path.join(save_dir, "debug_samples.json"), "w") as f:
        json.dump(debug_samples, f, indent=2)
        
    metadata = PuzzleDatasetMetadata(
        seq_len=max(config.max_seq_len, total_seq_len), 
        vocab_size=len(token_to_id), 
        pad_id=token_to_id["[PAD]"],
        ignore_label_id=token_to_id["[PAD]"], 
        blank_identifier_id=0, 
        num_puzzle_identifiers=1,
        total_puzzles=num_puzzles, 
        mean_puzzle_examples=1, 
        total_groups=num_puzzles, 
        sets=["all"],
        is_vision=True
    )
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
        
    print(f"\nGenerated {split} split: {num_puzzles} examples")

@cli.command(singleton=True)
def main(config: VisionDatasetConfig):
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    print("Generating Vision Q&A dataset...")
    
    token_to_id, id_to_token = build_vocabulary()
    
    generate_dataset(config, "train", token_to_id)
    generate_dataset(config, "test", token_to_id)
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token}, f, indent=2)
        
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "vision"], f)

    print("\n" + "="*50)
    print("Dataset generated successfully!")

if __name__ == "__main__":
    cli()
