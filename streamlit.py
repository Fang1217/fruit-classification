import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Map the class number to the class name
cat_to_name = {
    '0': 'Apple', 
    '1': 'Banana', 
    '2': 'Cherry', 
    '3': 'Chickoo', 
    '4': 'Grapes', 
    '5': 'Kiwi', 
    '6': 'Mango',
    '7': 'Orange', 
    '8': 'Strawberry'}

# Load the pre-trained VGG model with modification
vgg_model = models.vgg11(pretrained=True)
vgg_model.classifier[6] = torch.nn.Linear(vgg_model.classifier[6].in_features, 9)
vgg_model.load_state_dict(torch.load('./fruit_classification.pth', map_location=torch.device('cpu')))
vgg_model.eval()

st.title("Fruit Classification App")
st.write("This is a simple image classification web app to predict fruit classes.")

st.write("""Please upload an image file for the model to make predictions. The predicted fruit class will be displayed. 
         The model is trained on the following classes: Apple, Banana, Cherry, Chickoo, Grapes, Kiwi, Mango, Orange, and Strawberry.""")

# File uploader for image
image = st.file_uploader("Upload an image", type=["jpg", "png"])

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = vgg_model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    # Display the result
    st.write(f"Predicted class: {cat_to_name[str(predicted_class)]}")