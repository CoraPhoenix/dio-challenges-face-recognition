import streamlit as st
from PIL import Image
from io import BytesIO
import time
from utils.face_recognition import recognise_faces
from pathlib import Path

# helper function to enable Pillow image to be downloaded
def get_image_download_link(img):
    # Save the image to a BytesIO object
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # Change format if needed
    buffer.seek(0)
    return buffer

def main():

    # background image markdown code
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://static.vecteezy.com/system/resources/previews/022/416/212/non_2x/abstract-colorful-geometric-background-with-triangle-shape-pattern-and-molecular-vector.jpg");
    background-size: cover;
    }
    </style>
    '''

    # adding custom image to background
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Who is there?")

    st.text("A simple face recognition app which identifies people on a photo. If someone's not known, it \
        will label it as \"Unknown\". Give it a try!")
    
    # uploader bar
    uploaded_file = st.file_uploader("Select an image")

    # function call button
    if st.button("Detect faces!"):

        # if nothing is uploaded
        if not uploaded_file:
            st.error("Please upload an image first!")
        else:
            try:
                # reading image
                img = Image.open(BytesIO(uploaded_file.read()))

                st.write("Image uploaded successfully. Starting face detection...")
                start_time = time.perf_counter()
                detected_faces_img = recognise_faces(img, encodings_location = Path("utils/encodings.pkl")) # face recognition function 
                end_time = time.perf_counter()
                st.write("Faces detected successfully. Elapsed time: ", end_time - start_time, " seconds.")
                st.image(detected_faces_img, width = 400)

                if detected_faces_img:
                    st.write("If you want to save the image, click button below (page will be reloaded):")
                    buffer = get_image_download_link(detected_faces_img)
                    st.download_button(
                        label="Download image",
                        data=buffer,
                        file_name=uploaded_file.name
                    )
            
            except Exception as e:
                # if uploaded file is not an image
                print(e)
                st.error("Uploaded file must be an image file with the following formats: *.jpg, *.png, *.jpeg")
            
                

main()