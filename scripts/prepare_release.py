#!/usr/bin/env python3
import os
import shutil
import hashlib

def calculate_checksum(file_path):
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    # Create release directory
    os.makedirs("release", exist_ok=True)
    
    # Copy model checkpoint
    checkpoint_path = "checkpoints/last.ckpt"
    if os.path.exists(checkpoint_path):
        # Copy checkpoint to release directory
        release_path = "release/dog_breed_classifier.ckpt"
        shutil.copy2(checkpoint_path, release_path)
        
        # Calculate checksum
        checksum = calculate_checksum(release_path)
        
        # Create checksum file
        with open("release/checksum.txt", "w") as f:
            f.write(f"{checksum}  dog_breed_classifier.ckpt\n")
        
        print(f"Model checkpoint prepared for release:")
        print(f"File: {release_path}")
        print(f"SHA256: {checksum}")
        print("\nNext steps:")
        print("1. Create a new release on GitHub")
        print("2. Upload both files from the 'release' directory")
        print("3. Update the download URL in the README.md")
    else:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")

if __name__ == "__main__":
    main()
