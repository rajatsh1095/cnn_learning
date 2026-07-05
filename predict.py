import torch
from PIL import Image
from torchvision import transforms
from model import CatDogCNN

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load model
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load("./catdog_1_model.pth"))
    model.eval()

    #prepare image
    transform = transforms.Compose([
        transforms.Resize(224,224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    #predict
    with torch.no_grad():
        output = model(tensor)
        _,pred = torch.max(output, 1)
        classes = ["cat", "dog"]
        print(f"prediction: {classes[pred.item()]}")

if __name__ == "__main__":
    predict("./catdog_1_image.png")