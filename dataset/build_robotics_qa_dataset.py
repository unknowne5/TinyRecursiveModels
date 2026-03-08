"""
build_robotics_qa_dataset.py - Generate a visual Q&A and motor control dataset for a simple robotic arm.

This script creates a 2D simulation of a robotic arm with multiple joints (shoulder, elbow) 
and places various colored shapes in its workspace. It generates both perceptual questions 
(e.g., "what is the shoulder angle ?") and motor control tasks (e.g., "move to red square" 
outputting relative joint angle commands like "shoulder +15 elbow -20").
"""

import json
import math
import numpy as np
import os
import random
from typing import List, Tuple, Dict, Optional
from argdantic import ArgParser
from pydantic import BaseModel

from PIL import Image, ImageDraw
from common import PuzzleDatasetMetadata

cli = ArgParser()

class RoboticsDatasetConfig(BaseModel):
    output_dir: str = "data/robotics_qa_m1"
    max_seq_len: int = 32
    image_size: int = 64
    patch_size: int = 8
    num_train_puzzles: int = 8000
    num_test_puzzles: int = 1500
    seed: int = 42
    tasks: Optional[List[str]] = None

# --- Vocabulary and Tokenization ---

def build_vocabulary() -> Tuple[Dict[str, int], Dict[int, str]]:
    special_tokens = ["[PAD]", "[EOS]", "[SEP]", "[UNK]"]
    number_tokens = [str(i) for i in range(10)]  # 0-9
    color_tokens = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    shape_tokens = ["square", "circle", "triangle"]
    words = [
        "what", "is", "the", "shoulder", "elbow", "angle", "position", "?",
        "move", "to", "hand", "at", "grasp", "nothing", "+", "-"
    ]
    
    vocab = special_tokens + number_tokens + color_tokens + shape_tokens + words
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    return token_to_id, id_to_token

def custom_tokenizer(text: str) -> List[str]:
    tokens = []
    # simple tokenization that splits digits out for numbers
    text = text.replace("+", " + ").replace("-", " - ")
    for word in text.split():
        if word.lstrip('+-').isdigit():
            if word.startswith('+'):
                tokens.append('+')
                tokens.extend(list(word[1:]))
            elif word.startswith('-'):
                tokens.append('-')
                tokens.extend(list(word[1:]))
            else:
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

# --- Kinematics and Rendering ---

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255)
}

SHAPES = ["square", "circle", "triangle"]

def forward_kinematics(l1, l2, theta1, theta2, base_x, base_y):
    # theta1 is shoulder angle (0 is right, 90 is up)
    # theta2 is elbow angle relative to shoulder
    t1_rad = math.radians(theta1)
    t2_rad = math.radians(theta1 + theta2)
    
    elbow_x = base_x + l1 * math.cos(t1_rad)
    elbow_y = base_y - l1 * math.sin(t1_rad) # Y is inverted in image
    
    hand_x = elbow_x + l2 * math.cos(t2_rad)
    hand_y = elbow_y - l2 * math.sin(t2_rad)
    
    return (elbow_x, elbow_y), (hand_x, hand_y)

def inverse_kinematics(l1, l2, target_x, target_y, base_x, base_y):
    x = target_x - base_x
    y = base_y - target_y # Y is inverted
    
    # Distance to target
    dist = math.sqrt(x*x + y*y)
    if dist > l1 + l2:
        # Target out of reach
        return None
        
    # Law of cosines
    cos_angle2 = (x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2)
    cos_angle2 = max(-1.0, min(1.0, cos_angle2))
    theta2_rad = math.acos(cos_angle2)
    
    k1 = l1 + l2 * math.cos(theta2_rad)
    k2 = l2 * math.sin(theta2_rad)
    theta1_rad = math.atan2(y, x) - math.atan2(k2, k1)
    
    return int(math.degrees(theta1_rad)), int(math.degrees(theta2_rad))

def generate_environment(image_size: int, allowed_tasks: List[str]):
    # Arm configuration
    base_x, base_y = image_size // 2, image_size - 10
    l1, l2 = image_size // 3, image_size // 3
    
    # Current joint angles
    current_t1 = random.randint(30, 150)
    current_t2 = random.randint(-120, 120)
    
    elbow_pos, hand_pos = forward_kinematics(l1, l2, current_t1, current_t2, base_x, base_y)
    
    # Generate objects
    objects = []
    num_objects = random.randint(1, 3)
    
    # Try to place objects within reach but avoiding overlaps
    for i in range(num_objects):
        shape = random.choice(SHAPES)
        color = random.choice(list(COLORS.keys()))
        
        placed = False
        attempts = 0
        while not placed and attempts < 20:
            # For 25% of the environments, intentionally place the first object right at the hand
            if i == 0 and random.random() < 0.25:
                target_t1 = current_t1
                target_t2 = current_t2
            else:
                target_t1 = random.randint(30, 150)
                target_t2 = random.randint(-120, 120)
            _, obj_pos = forward_kinematics(l1, l2, target_t1, target_t2, base_x, base_y)
            
            # Check overlap
            overlap = False
            for obj in objects:
                d = math.hypot(obj_pos[0] - obj['x'], obj_pos[1] - obj['y'])
                if d < 10:
                    overlap = True
            
            if not overlap:
                objects.append({
                    'x': int(obj_pos[0]), 
                    'y': int(obj_pos[1]), 
                    'shape': shape, 
                    'color': color,
                    't1': target_t1,
                    't2': target_t2
                })
                placed = True
            attempts += 1
            
    # Draw image
    img = Image.new('RGB', (image_size, image_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw objects
    for obj in objects:
        r = 4
        bbox = [obj['x']-r, obj['y']-r, obj['x']+r, obj['y']+r]
        color_rgb = COLORS[obj['color']]
        if obj['shape'] == 'square':
            draw.rectangle(bbox, fill=color_rgb)
        elif obj['shape'] == 'circle':
            draw.ellipse(bbox, fill=color_rgb)
        elif obj['shape'] == 'triangle':
            draw.polygon([
                (obj['x'], obj['y']-r),
                (obj['x']-r, obj['y']+r),
                (obj['x']+r, obj['y']+r)
            ], fill=color_rgb)
            
    # Draw arm (white)
    arm_color = (255, 255, 255)
    draw.line([(base_x, base_y), (int(elbow_pos[0]), int(elbow_pos[1]))], fill=arm_color, width=2)
    draw.line([(int(elbow_pos[0]), int(elbow_pos[1])), (int(hand_pos[0]), int(hand_pos[1]))], fill=arm_color, width=2)
    
    # Joints
    draw.ellipse([base_x-2, base_y-2, base_x+2, base_y+2], fill=(128,128,128))
    draw.ellipse([int(elbow_pos[0])-2, int(elbow_pos[1])-2, int(elbow_pos[0])+2, int(elbow_pos[1])+2], fill=(128,128,128))
    draw.ellipse([int(hand_pos[0])-2, int(hand_pos[1])-2, int(hand_pos[0])+2, int(hand_pos[1])+2], fill=(255,0,0)) # red hand
    
    image_np = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    
    target_obj_idx = None
    
    # Generate task
    if not allowed_tasks:
        raise ValueError("allowed_tasks list cannot be empty")
        
    task_type = random.choice(allowed_tasks)
    
    if task_type == "qa_state":
        if random.random() < 0.5:
            question = "what is the shoulder angle ?"
            answer = str(current_t1)
        else:
            question = "what is the elbow angle ?"
            answer = str(current_t2)
            
    elif task_type == "qa_vision":
        # Check what is at hand
        at_hand = None
        for obj in objects:
            if math.hypot(obj['x'] - hand_pos[0], obj['y'] - hand_pos[1]) < 8:
                at_hand = obj
                break
        
        question = "what is at the hand ?"
        if at_hand:
            answer = f"{at_hand['color']} {at_hand['shape']}"
        else:
            answer = "nothing"
            
    else: # motor_control
        if len(objects) > 0:
            target_obj_idx = random.randint(0, len(objects) - 1)
            target_obj = objects[target_obj_idx]
            question = f"move to {target_obj['color']} {target_obj['shape']}"
            
            delta_t1 = target_obj['t1'] - current_t1
            delta_t2 = target_obj['t2'] - current_t2
            
            dt1_str = f"+{delta_t1}" if delta_t1 >= 0 else str(delta_t1)
            dt2_str = f"+{delta_t2}" if delta_t2 >= 0 else str(delta_t2)
            
            answer = f"shoulder {dt1_str} elbow {dt2_str}"
        else:
            question = "what is the shoulder angle ?"
            answer = str(current_t1)
            
    # Generate solution image if needed
    solution_img = None
    if task_type == "motor_control" and target_obj_idx is not None:
        sol_img = Image.new('RGB', (image_size, image_size), color=(0, 0, 0))
        sol_draw = ImageDraw.Draw(sol_img)
        
        # Draw objects
        for obj in objects:
            r = 4
            bbox = [obj['x']-r, obj['y']-r, obj['x']+r, obj['y']+r]
            color_rgb = COLORS[obj['color']]
            if obj['shape'] == 'square':
                sol_draw.rectangle(bbox, fill=color_rgb)
            elif obj['shape'] == 'circle':
                sol_draw.ellipse(bbox, fill=color_rgb)
            elif obj['shape'] == 'triangle':
                sol_draw.polygon([
                    (obj['x'], obj['y']-r),
                    (obj['x']-r, obj['y']+r),
                    (obj['x']+r, obj['y']+r)
                ], fill=color_rgb)
                
        # Draw target arm
        target_obj = objects[target_obj_idx]
        target_t1, target_t2 = target_obj['t1'], target_obj['t2']
        t_elbow_pos, t_hand_pos = forward_kinematics(l1, l2, target_t1, target_t2, base_x, base_y)
        
        arm_color = (255, 255, 255)
        sol_draw.line([(base_x, base_y), (int(t_elbow_pos[0]), int(t_elbow_pos[1]))], fill=arm_color, width=2)
        sol_draw.line([(int(t_elbow_pos[0]), int(t_elbow_pos[1])), (int(t_hand_pos[0]), int(t_hand_pos[1]))], fill=arm_color, width=2)
        
        sol_draw.ellipse([base_x-2, base_y-2, base_x+2, base_y+2], fill=(128,128,128))
        sol_draw.ellipse([int(t_elbow_pos[0])-2, int(t_elbow_pos[1])-2, int(t_elbow_pos[0])+2, int(t_elbow_pos[1])+2], fill=(128,128,128))
        sol_draw.ellipse([int(t_hand_pos[0])-2, int(t_hand_pos[1])-2, int(t_hand_pos[0])+2, int(t_hand_pos[1])+2], fill=(255,0,0))
        
        solution_img = np.array(sol_img).transpose(2, 0, 1).astype(np.float32) / 255.0

    return image_np, question, answer, solution_img

# --- Main Dataset Generation ---

def generate_dataset(config: RoboticsDatasetConfig, split: str, token_to_id: Dict[str, int]):
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
    total_seq_len = config.max_seq_len + seq_len_patches
    
    for i in range(num_puzzles):
        if i % 1000 == 0:
            print(f"  Generating {split} example {i}/{num_puzzles}")
            
        image, question, answer, solution_img = generate_environment(config.image_size, config.tasks)
        
        txt_inp = encode_text(question, token_to_id, max_len=config.max_seq_len, add_eos=False)
        lbl = encode_text(answer, token_to_id, max_len=total_seq_len, add_eos=True)
        
        if i < 20:
            debug_samples.append({
                "id": i,
                "question": question,
                "answer": answer,
                "encoded_text_inputs": str(txt_inp.tolist()),
                "encoded_labels": str(lbl.tolist()),
            })
            
        img_uint8 = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_filename = f"image_{i}.png"
        img_pil.save(os.path.join(images_dir, img_filename))
        
        if solution_img is not None and i < 20:
            sol_uint8 = (solution_img.transpose(1, 2, 0) * 255).astype(np.uint8)
            sol_pil = Image.fromarray(sol_uint8)
            sol_filename = f"image_{i}_solution.png"
            sol_pil.save(os.path.join(images_dir, sol_filename))
            
        all_inputs.append(i)
        all_text_inputs.append(txt_inp)
        all_labels.append(lbl)
        puzzle_indices.append(len(all_inputs))
        puzzle_identifiers.append(0)
        group_indices.append(i + 1)
    
    results_np = {
        "inputs": np.array(all_inputs, dtype=np.int32),
        "text_inputs": np.array(all_text_inputs, dtype=np.int32),
        "labels": np.array(all_labels, dtype=np.int32),
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
def main(config: RoboticsDatasetConfig):
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    if config.tasks is None:
        config.tasks = ["qa_state", "qa_vision", "motor_control"]

    print("Generating Robotics Q&A and Motor Control dataset...")
    
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
