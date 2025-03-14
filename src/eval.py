import os
import torch
import pytorch_lightning as pl
from datamodules.dog_breed_datamodule import DogBreedDataModule
from train import DogBreedClassifier
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate Dog Breed Classifier')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/dog_breed/dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Initialize data module
    data_module = DogBreedDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    data_module.setup()

    # Load model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(
        args.checkpoint_path,
        num_classes=len(data_module.train_dataset.dataset.classes)
    )
    model.eval()

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
    )

    # Run evaluation
    results = trainer.validate(model=model, datamodule=data_module)
    
    print("\nValidation Results:")
    for metric_name, value in results[0].items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
