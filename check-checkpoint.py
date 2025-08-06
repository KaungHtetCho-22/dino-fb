import torch
import numpy as np

# Method 1: Set weights_only=False (since you trust your own checkpoint)
print("Loading checkpoint with weights_only=False...")
try:
    checkpoint = torch.load('./dino_outputs_leukemia_patch8/checkpoint0060.pth', weights_only=False)
    print("Checkpoint loaded successfully!")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit()

print("\n" + "="*50)
print("CHECKPOINT ANALYSIS")
print("="*50)

# Print main keys
print("\nMain keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Analyze the student model (main model)
if 'student' in checkpoint:
    print("\nStudent model structure:")
    student_weights = checkpoint['student']
    
    # Group keys by component
    backbone_keys = []
    head_keys = []
    other_keys = []
    
    for key in student_weights.keys():
        if key.startswith('head'):
            head_keys.append(key)
        elif any(component in key for component in ['patch_embed', 'pos_embed', 'blocks', 'norm']):
            backbone_keys.append(key)
        else:
            other_keys.append(key)
    
    print(f" Backbone components ({len(backbone_keys)} keys):")
    for key in backbone_keys[:10]:  # Show first 10
        if hasattr(student_weights[key], 'shape'):
            print(f"    - {key}: {student_weights[key].shape}")
        else:
            print(f"    - {key}: {type(student_weights[key])}")
    if len(backbone_keys) > 10:
        print(f"    ... and {len(backbone_keys) - 10} more backbone keys")
    
    print(f"\nHead components ({len(head_keys)} keys):")
    for key in head_keys:
        if hasattr(student_weights[key], 'shape'):
            print(f"    - {key}: {student_weights[key].shape}")
        else:
            print(f"    - {key}: {type(student_weights[key])}")
    
    if other_keys:
        print(f"\nOther components ({len(other_keys)} keys):")
        for key in other_keys:
            if hasattr(student_weights[key], 'shape'):
                print(f"    - {key}: {student_weights[key].shape}")
            else:
                print(f"    - {key}: {type(student_weights[key])}")

# Check teacher model if exists
if 'teacher' in checkpoint:
    print(f"\nTeacher model: {len(checkpoint['teacher'])} parameters")

# Check training info
if 'epoch' in checkpoint:
    print(f"\nTraining info:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    
if 'args' in checkpoint:
    print(f"  - Architecture: {getattr(checkpoint['args'], 'arch', 'Unknown')}")
    print(f"  - Patch size: {getattr(checkpoint['args'], 'patch_size', 'Unknown')}")
    print(f"  - Image size: {getattr(checkpoint['args'], 'img_size', 'Unknown')}")

print("\n" + "="*50)
print("BACKBONE EXTRACTION PREVIEW")
print("="*50)

# Extract backbone weights (what we'll use for ViTDet)
if 'student' in checkpoint:
    student_weights = checkpoint['student']
    backbone_weights = {k: v for k, v in student_weights.items() if not k.startswith('head')}
    
    print(f"Extracted {len(backbone_weights)} backbone parameters")
    print("Key backbone components:")
    
    key_components = ['patch_embed.proj.weight', 'pos_embed', 'blocks.0.norm1.weight', 'norm.weight']
    for component in key_components:
        if component in backbone_weights:
            print(f"{component}: {backbone_weights[component].shape}")
        else:
            print(f"{component}: Not found")

print("\n" + "="*50)
print("VITDET COMPATIBILITY CHECK")
print("="*50)

# Check ViT architecture details
if 'student' in checkpoint:
    student_weights = checkpoint['student']
    
    # Check embedding dimension
    if 'patch_embed.proj.weight' in student_weights:
        embed_weight = student_weights['patch_embed.proj.weight']
        embed_dim = embed_weight.shape[0]
        print(f"Embedding dimension: {embed_dim}")
        
        # Determine ViT size
        if embed_dim == 384:
            vit_size = "Small"
        elif embed_dim == 768:
            vit_size = "Base"
        elif embed_dim == 1024:
            vit_size = "Large"
        else:
            vit_size = "Custom"
        print(f"ViT Size: {vit_size}")
    
    # Check number of transformer blocks
    block_keys = [k for k in student_weights.keys() if k.startswith('blocks.')]
    if block_keys:
        block_numbers = [int(k.split('.')[1]) for k in block_keys if k.split('.')[1].isdigit()]
        num_layers = max(block_numbers) + 1 if block_numbers else 0
        print(f"Number of layers: {num_layers}")
    
    # Check positional embeddings
    if 'pos_embed' in student_weights:
        pos_embed_shape = student_weights['pos_embed'].shape
        print(f"Positional embeddings: {pos_embed_shape}")
        # Calculate implied patch size and image size
        num_patches = pos_embed_shape[1] - 1  # -1 for cls token
        # This is approximate, actual calculation depends on your training setup

print("\nReady for ViTDet integration!")
print("Use the 'backbone_weights' extracted above for transfer learning.")