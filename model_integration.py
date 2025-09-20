import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# --- Face Detector ---
class FaceDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # Pass the input through feature extractor and classifier
        x = self.features(x)
        return self.classifier(x)


# --- SE Block ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Define the fully connected layers for SE block
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        # Apply SE block: global average pooling, fully connected layers, and scaling
        se = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(x.size(0), x.size(1), 1, 1)
        return x * se


# --- DeepFake Model using EfficientNet ---
class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        # Load EfficientNet backbone
        backbone = models.efficientnet_b0(pretrained=True)
        # Freeze all layers except the last 5
        for param in backbone.features[-5:].parameters():
            param.requires_grad = True

        self.feature_extractor = backbone.features
        self.se_block = SEBlock(1280)  # SE block
        self.pool = nn.AdaptiveAvgPool2d(1)  # Adaptive average pooling
        self.classifier = nn.Linear(1280, 2)  # Classifier for real vs deepfake

    def forward(self, x):
        # Pass input through feature extractor, SE block, and classifier
        x = self.feature_extractor(x)
        x = self.se_block(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --- Unified Model ---
class UnifiedFaceDeepFakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize face detector model
        self.face_detector = FaceDetectorCNN()
        # Initialize EfficientNet backbone for deepfake classification
        backbone = models.efficientnet_b0(pretrained=True)
        self.df_feature_extractor = backbone.features
        self.se_block = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.df_classifier = nn.Linear(1280, 2)  # 2 classes: real (1) or deepfake (0)

    def forward(self, face_tensor, df_tensor):
        # Run face detection first
        face_score = self.face_detector(face_tensor)
        if face_score.item() <= 0.5:
            # No face detected, return -1 to indicate no face
            return torch.tensor([-1])

        # If face detected, proceed with deepfake model
        features = self.df_feature_extractor(df_tensor)
        features = self.se_block(features)
        pooled = self.pool(features).view(features.size(0), -1)
        logits = self.df_classifier(pooled)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1)  # 0 = DeepFake, 1 = Real
        return prediction


# --- Transforms ---
# Transform for the face detection model (resize and normalize)
face_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Transform for the deepfake detection model (resize and normalize)
df_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Load Model ---
# Set device for model (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedFaceDeepFakeModel().to(device)

# Load the pre-trained model weights
# Make sure to provide paths to the saved model weights
model.face_detector.load_state_dict(torch.load("faceclassifier.pth"))
model.df_feature_extractor.load_state_dict(torch.load("deepfake.pth"), strict=False)

# Set the model to evaluation mode (disables dropout layers)
model.eval()

# --- Inference Function ---
def run_inference(image_path):
    """
    Run inference on the given image to detect whether it contains a real face or a deepfake.
    :param image_path: Path to the image file.
    """
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")

    # Transform the image for the face detection model and deepfake model
    face_tensor = face_transform(image).unsqueeze(0).to(device)
    df_tensor = df_transform(image).unsqueeze(0).to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        result = model(face_tensor, df_tensor).item()

    # Print the result of the inference
    if result == -1:
        print("No face detected.")
    elif result == 1:
        print("Real face detected.")
    else:
        print("DeepFake detected.")


# --- Save Unified Model ---
def save_model(model, path="unified_model_final.pth"):
    """
    Save the model state dictionary to a file.
    :param model: The model to be saved.
    :param path: The file path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")

# Save the model
save_model(model)

# --- Test ---
# Test the model with an image
run_inference("test.jpg")