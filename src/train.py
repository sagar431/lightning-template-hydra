import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules.dog_breed_datamodule import DogBreedDataModule
import timm

class DogBreedClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def main():
    # Data module
    data_module = DogBreedDataModule(
        data_dir="data/dog_breed/dataset",
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    # Model
    model = DogBreedClassifier(
        num_classes=len(data_module.train_dataset.dataset.classes),
        learning_rate=1e-3
    )
    
    # Logger
    logger = TensorBoardLogger("lightning_logs", name="dog_breed")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='dog-breed-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_last=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    # Train
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()