import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from train import DogBreedClassifier
from datamodules.dog_breed_datamodule import DogBreedDataModule
import random
import matplotlib.pyplot as plt

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def plot_predictions(images, true_classes, predicted_classes, confidences, num_cols=3):
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, (img, true_class, pred_class, conf) in enumerate(zip(images, true_classes, predicted_classes, confidences)):
        axes[idx].imshow(img)
        color = 'green' if true_class == pred_class else 'red'
        title = f'True: {true_class}\nPred: {pred_class}\nConf: {conf:.1%}'
        axes[idx].set_title(title, color=color)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Run inference on Dog Breed Classifier')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/dog_breed/dataset', help='Path to dataset')
    parser.add_argument('--num_images', type=int, default=9, help='Number of random images to run inference on')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save prediction plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data module to get class names
    data_module = DogBreedDataModule(data_dir=args.data_dir)
    data_module.setup()
    classes = data_module.train_dataset.dataset.classes

    # Load model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(
        args.checkpoint_path,
        num_classes=len(classes)
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get transform
    transform = get_transform()

    # Get all image paths
    image_paths = []
    for class_name in classes:
        class_dir = os.path.join(args.data_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((os.path.join(class_dir, img_name), class_name))

    # Randomly select images
    selected_images = random.sample(image_paths, min(args.num_images, len(image_paths)))

    print(f"\nRunning inference on {len(selected_images)} images...")
    print("-" * 50)

    # Run inference
    correct = 0
    images = []
    true_classes = []
    predicted_classes = []
    confidences = []
    
    for img_path, true_class in selected_images:
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()

        predicted_class = classes[predicted_idx]
        is_correct = predicted_class == true_class
        correct += int(is_correct)
        
        # Store results for plotting
        images.append(image)
        true_classes.append(true_class)
        predicted_classes.append(predicted_class)
        confidences.append(confidence)
        
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"True Class: {true_class}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
    
    # Plot and save predictions
    fig = plot_predictions(images, true_classes, predicted_classes, confidences)
    plt.savefig(os.path.join(args.output_dir, 'predictions.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nAccuracy on {len(selected_images)} random images: {correct/len(selected_images):.2%}")
    print(f"Predictions plot saved to {os.path.join(args.output_dir, 'predictions.png')}")

if __name__ == "__main__":
    main()