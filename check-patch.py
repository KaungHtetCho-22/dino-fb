import torch

# Load the checkpoint
checkpoint = torch.load('dino_outputs_leukemia_patch4/checkpoint0060.pth', 
                       map_location='cpu', weights_only=False)

print("Top-level keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\n" + "="*50)

# DINO checkpoints typically have 'student' and 'teacher' keys
if 'student' in checkpoint:
    print("\nStudent model keys (first 20):")
    student_keys = list(checkpoint['student'].keys())
    for key in student_keys[:20]:
        print(f"  {key}")
    
    # Look for patch embedding in student
    print("\nLooking for patch embedding in student:")
    for key in student_keys:
        if 'patch' in key.lower():
            print(f"  Found: {key} -> {checkpoint['student'][key].shape}")

if 'teacher' in checkpoint:
    print("\nTeacher model keys (first 20):")
    teacher_keys = list(checkpoint['teacher'].keys())
    for key in teacher_keys[:20]:
        print(f"  {key}")
    
    # Look for patch embedding in teacher
    print("\nLooking for patch embedding in teacher:")
    for key in teacher_keys:
        if 'patch' in key.lower():
            print(f"  Found: {key} -> {checkpoint['teacher'][key].shape}")

# Also check if there's a direct state dict
if isinstance(checkpoint, dict) and 'student' not in checkpoint and 'teacher' not in checkpoint:
    print("\nDirect state dict keys (first 20):")
    all_keys = list(checkpoint.keys())
    for key in all_keys[:20]:
        print(f"  {key}")
    
    print("\nLooking for patch-related keys:")
    for key in all_keys:
        if 'patch' in key.lower():
            print(f"  Found: {key} -> {checkpoint[key].shape}")