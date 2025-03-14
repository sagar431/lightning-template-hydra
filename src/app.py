import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import base64
from train import DogBreedClassifier
from torchvision import transforms

app = FastAPI(title="Dog Breed Classifier")

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Set up templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Initialize model
checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints/last.ckpt")

# Define class names
classes = [
    'Beagle', 'Boxer', 'Bulldog', 'Chihuahua', 'Dachshund',
    'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever',
    'Poodle', 'Rottweiler'
]

# Load model
model = DogBreedClassifier.load_from_checkpoint(
    checkpoint_path,
    num_classes=len(classes)
)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_prediction(image):
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_5_prob, top_5_idx = torch.topk(probabilities, 5)
        
    # Get top 5 predictions
    predictions = []
    for idx, (prob, class_idx) in enumerate(zip(top_5_prob, top_5_idx)):
        predictions.append({
            'rank': idx + 1,
            'breed': classes[class_idx],
            'probability': float(prob) * 100
        })
    
    return predictions

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and validate image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Get predictions
    predictions = get_prediction(image)
    
    # Convert image to base64 for display
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "predictions": predictions,
        "image": img_str
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
