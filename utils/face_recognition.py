import pickle as pkl
import face_recognition
from PIL import Image, ImageDraw
from pathlib import Path
from collections import Counter
from utils.config import Config
import numpy as np

# non-public function
def _recognise_face(encoding, encoding_dict):
    """Compares the face encoding with the existing set and returns the name with
    highest probability"""
    boolean_matches = face_recognition.compare_faces(
        encoding_dict["encodings"], encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, encoding_dict["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(draw, bounding_box, name):

    """Displays bounding box around the detected image with its label"""
    
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=Config.BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="green",
        outline="green",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

def recognise_faces(image, model = "hog", encodings_location = Path("encodings.pkl")):

    """Detects each face in an given image and compares them to the existing reference
    database. Returns the label for each face and its bounding box"""

    with encodings_location.open(mode="rb") as f:
        encoding_dict = pkl.load(f)

    # loading image
    if type(image) is str:
        input_img = face_recognition.load_image_file(image)
    else:
        input_img = np.array(image.convert("RGB"))

    # getting detected faces from image and encoding them
    input_face_locations = face_recognition.face_locations(
        input_img, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_img, input_face_locations
    )

    # drawing image
    pillow_image = Image.fromarray(input_img)
    draw = ImageDraw.Draw(pillow_image)

    # comparing encoded face with all previously encoded faces

    for bounding_box, encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognise_face(encoding, encoding_dict) # getting detected face and its location
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    # show image with bounding boxes
    del draw
    
    return pillow_image
    