import torch
import numpy as np
from collections import OrderedDict

def extract_dino_backbone_weights(checkpoint_path, save_path=None):
    """
    Extract backbone weights from DINO checkpoint and prepare for ViTDet
    """
    print(f"Loading DINO checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    if 'student' not in checkpoint:
        raise ValueError("No 'student' key found in checkpoint")
    
    student_weights = checkpoint['student']
    
    # Remove 'module.' prefix (from DataParallel) and extract backbone
    clean_weights = OrderedDict()
    
    for key, value in student_weights.items():
        # Remove 'module.' prefix
        clean_key = key.replace('module.', '')
        
        # Keep only backbone weights (exclude head)
        if clean_key.startswith('backbone.'):
            # Remove 'backbone.' prefix to match ViT structure
            vit_key = clean_key.replace('backbone.', '')
            clean_weights[vit_key] = value
        elif clean_key.startswith('head.'):
            # Skip head weights - we don't need them for ViTDet
            continue
    
    print(f"Extracted {len(clean_weights)} backbone parameters")
    
    # Print key information
    print("\nArchitecture Analysis:")
    
    # Check embedding dimension
    if 'patch_embed.proj.weight' in clean_weights:
        embed_weight = clean_weights['patch_embed.proj.weight']
        embed_dim = embed_weight.shape[0]
        patch_size = embed_weight.shape[2]  # Assuming square patches
        print(f"  - Embedding dimension: {embed_dim}")
        print(f"  - Patch size: {patch_size}")
        
        # Determine ViT size
        if embed_dim == 384:
            vit_size = "Small"
        elif embed_dim == 768:
            vit_size = "Base"  
        elif embed_dim == 1024:
            vit_size = "Large"
        else:
            vit_size = "Custom"
        print(f"  - ViT variant: {vit_size}")
    
    # Check positional embeddings
    if 'pos_embed' in clean_weights:
        pos_embed_shape = clean_weights['pos_embed'].shape
        print(f"  - Positional embeddings shape: {pos_embed_shape}")
        num_patches = pos_embed_shape[1] - 1  # -1 for cls token
        print(f"  - Number of patches: {num_patches}")
        # For patch_size=8, if we have 784 patches, image size would be 224x224
        # sqrt(784) = 28, so 28*8 = 224
        import math
        if num_patches > 0:
            patches_per_side = int(math.sqrt(num_patches))
            implied_img_size = patches_per_side * patch_size
            print(f"  - Implied image size: {implied_img_size}x{implied_img_size}")
    
    # Count transformer blocks
    block_keys = [k for k in clean_weights.keys() if k.startswith('blocks.')]
    if block_keys:
        block_numbers = [int(k.split('.')[1]) for k in block_keys if len(k.split('.')) > 1 and k.split('.')[1].isdigit()]
        num_layers = max(block_numbers) + 1 if block_numbers else 0
        print(f"  - Number of transformer layers: {num_layers}")
    
    # Check attention heads
    if 'blocks.0.attn.qkv.weight' in clean_weights:
        qkv_weight = clean_weights['blocks.0.attn.qkv.weight']
        # qkv_weight shape is [3*num_heads*head_dim, embed_dim]
        total_qkv_dim = qkv_weight.shape[0]
        num_heads = total_qkv_dim // (3 * embed_dim)
        print(f"  - Number of attention heads: {num_heads}")
        print(f"  - Head dimension: {embed_dim // num_heads}")
    
    # Print some key weights for verification
    print(f"\nKey backbone components:")
    key_components = ['patch_embed.proj.weight', 'pos_embed', 'cls_token', 'blocks.0.norm1.weight', 'norm.weight']
    for component in key_components:
        if component in clean_weights:
            shape = clean_weights[component].shape
            print(f"{component}: {shape}")
        else:
            print(f"{component}: Not found")
    
    # Save cleaned weights if path provided
    if save_path:
        save_dict = {
            'backbone_weights': clean_weights,
            'model_info': {
                'architecture': 'vit_small',
                'patch_size': patch_size if 'patch_embed.proj.weight' in clean_weights else 8,
                'embed_dim': embed_dim if 'patch_embed.proj.weight' in clean_weights else 384,
                'num_layers': num_layers if block_keys else 12,
                'num_heads': num_heads if 'blocks.0.attn.qkv.weight' in clean_weights else 6,
            },
            'original_epoch': checkpoint.get('epoch', 'unknown'),
            'original_args': checkpoint.get('args', None)
        }
        
        torch.save(save_dict, save_path)
        print(f"\nSaved cleaned weights to: {save_path}")
    
    return clean_weights

def verify_weight_compatibility(backbone_weights):
    """
    Verify if the extracted weights are compatible with ViTDet
    """
    print("\nViTDet Compatibility Check:")
    
    required_components = [
        'patch_embed.proj.weight',
        'pos_embed', 
        'cls_token',
        'blocks.0.norm1.weight',
        'blocks.0.attn.qkv.weight'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in backbone_weights:
            missing_components.append(component)
    
    if not missing_components:
        print("All required components present!")
        print("Ready for ViTDet integration!")
        return True
    else:
        print("Missing components:")
        for component in missing_components:
            print(f"    - {component}")
        return False

if __name__ == "__main__":
    # Extract weights from your DINO checkpoint
    checkpoint_path = './checkpoint0080.pth'
    save_path = './dino_backbone_weights_for_vitdet.pth'
    
    try:
        backbone_weights = extract_dino_backbone_weights(checkpoint_path, save_path)
        compatibility = verify_weight_compatibility(backbone_weights)
        
        if compatibility:
            print(f"\nSuccess! Your DINO weights are ready for ViTDet.")
            print(f"Cleaned weights saved to: {save_path}")
            print(f"\nNext steps:")
            print(f"1. Set up ViTDet repository")
            print(f"2. Create custom config for ViT-Small with patch_size=8")
            print(f"3. Load these weights into ViTDet backbone")
            print(f"4. Fine-tune on your object detection dataset")
        else:
            print(f"\nSome compatibility issues found. Please check the missing components.")
            
    except Exception as e:
        print(f"Error: {e}")