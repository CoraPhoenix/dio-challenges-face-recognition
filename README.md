# dio-challenges-face-recognition
Project made during DIO's "BairesDev - Machine Learning Practitioner" Bootcamp, under the section "Criando um Sistema de Reconhecimento Facial do Zero"

# Description

In this project, I made a simple app using Streamlit which implements a face recognition program. This program was trained using the [Kaggle's LFW People Dataset](https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data) and was based on the project shown at [this article](https://realpython.com/face-recognition-with-python/#step-1-prepare-your-environment-and-data).

# App Layout

This is how it looks like:

**(image to be inserted here)**

# How to use the app

1. Upload image
2. Click on "Detect faces!" button
3. The image with bounding boxes and labels for each box will appear right below

**(image to be inserted)**

4. If you want to download the image with the boxes, click on "Download image" (notice this will reload the app)

# Development Process

In order to create the app, I followed the steps below:

1. Using the dataset to create the encodings file (this was done on [this Kaggle notebook](https://www.kaggle.com/code/christophercamilo/face-detection-and-recognition))
2. Creating the app layout using Streamlit
3. Using the trained encodings file to create the face recognition program
4. Implementing the face recognition program on the app

# Known Issues and Constraints

- The model relies on an encodings file, which means it will need to be recreated in case of adding more people to the dataset
- The used images in the dataset contain some noise ("borders" around faces used in the training process), which affects the program performance

# Proposed Next Steps

1. Crop each image in the training set so that it becomes more robust
2. Adding more images of each person to the training set
