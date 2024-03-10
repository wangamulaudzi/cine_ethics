# Import statements
import cv2
import dlib
from google.cloud import storage
import io
import face_recognition
import numpy as np
import os
import pickle
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN
import random
from dotenv import load_dotenv

#loading credentials
load_dotenv()

# Get the path to the key file from the environment variable
key_file_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Authenticate with Google Cloud using the key file
client = storage.Client.from_service_account_json(key_file_path)

bucket_name = "cine_ethics"
bucket = client.get_bucket(bucket_name)

table_path = os.getenv("CINE_PICK_TABLE")

synopsis_screengrabs_df = pd.read_csv(table_path)

def movie_to_analyse(title):
    """
    Function to get all the characters from the movie title

    Returns:
        List of images with all the characters
    """
    movie_info = synopsis_screengrabs_df[synopsis_screengrabs_df["title"] == title.lower()]

    # Path to the movie on the bucket
    movie_path = movie_info["paths"]

    # Store the image path names
    faces_list_paths = []
    faces_image_arrays = []

    # Load the pre-trained face detector model from Dlib
    detector = dlib.get_frontal_face_detector()

    faces_list = [] # List to store the images that have faces

    data = [] # List to store the encodings

    # Get blobs within the movie folder
    blobs = bucket.list_blobs(prefix=movie_path)

    # Get a random sample of blobs
    sample_size = 200 # Number of images to sample as opposed to the full 1k
    random_sample = random.sample(list(blobs), sample_size)

    for blob in random_sample:
        # Download the image as bytes
        img_bytes = blob.download_as_bytes()

        # Open the image
        img = Image.open(io.BytesIO(img_bytes))

        # Convert image to an array
        img = np.array(img)

        # Convert image to grayscale (required for Haar cascades)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray)

        # If image is detected as a face
        if len(faces) > 0:

            # Append the image with detected faces
            faces_list.append(img)

            # Append image path
            faces_list_paths.append(blob.name)

            # Append array image to list
            faces_image_arrays.append(img)

            # Convert image to colour scale
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, model="hog")

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # build a dictionary of the image path, bounding box location,
            # and facial encodings for the current image
            d = [{"imagePath": blob.name, "loc": box, "encoding": enc}
                for (box, enc) in zip(boxes, encodings)]
            data.extend(d)

    # Write the encodings to disk
    path_encodings = os.path.join("raw_data/deep_face_encodings/", movie_info["title"].values[0])
    path_encodings = str(path_encodings)

    # Ensure that the directory exists
    os.makedirs(str(path_encodings), exist_ok=True)

    # Dump the facial encodings data to disk
    file_name = movie_info["title"].values[0] + ".pickle"
    file_name = str(file_name)

    f = open(os.path.join(path_encodings, file_name), "wb")
    f.write(pickle.dumps(data))
    f.close()

    # Load the serialized face encodings + bounding box locations from
    # disk, then extract the set of encodings to so we can cluster on them
    encodings_path = os.path.join(path_encodings, file_name)

    #print("Loading encodings at", encodings_path, "...")
    data = pickle.loads(open(encodings_path, "rb").read())
    data = np.array(data)

    encodings = [d["encoding"] for d in data]

    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)

    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)

    grouped_faces = [] # Numpy arrays for each set of grouped faces
    grouped_imread_faces = [] # Corresponding imread objects
    grouped_bounding_boxes = [] # Corresponding bounding boxes

    # Loop over the unique face integers
    for labelID in labelIDs:
        # Find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes from the set
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

        # Initialize the list of faces to include in the montage
        faces = []

        # Initialize list to store all the coloured images for each character
        # These images will be used for the characters the user picks to morph
        all_faces_for_morph = []

        # List to store bounding boxes for each face
        bounding_box_faces = []

        # loop over the sampled indexes
        for i in idxs:
            # Load the input image from Google Cloud Storage and extract the face ROI
            blob = client.bucket(bucket_name).blob(data[i]["imagePath"])
            image_bytes = blob.download_as_string()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            image = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

            # Append the coloured image for later analysis
            all_faces_for_morph.append(image)

            (top, right, bottom, left) = data[i]["loc"]

            # Bounding boxes of the image
            face = image[top:bottom, left:right]
            bounding_box_faces.append(face)

            # Add to the faces montage list
            faces.append(image)

        grouped_faces.append(faces)
        grouped_imread_faces.append(all_faces_for_morph)
        grouped_bounding_boxes.append(bounding_box_faces)

    # List to store one random photo for each character
    random_faces = []

    # Lists to store the random photo's imread object and facial bounds
    random_face_imread_list = []
    random_face_bounds_list = []

    # Return one of each in grouped faces
    for i in range(len(grouped_faces)):
        random_face = random.sample(grouped_faces[i], 1)
        random_faces.append(random_face)

        # Get the index of this random face
        # Finding index where grouped_faces[i] == random_face[0]
        indx_random_face = np.where([np.array_equal(arr, random_face[0]) for arr in grouped_faces[i]])[0][0]

        # Select the random face's imread object
        random_face_imread = grouped_imread_faces[i][indx_random_face]
        random_face_imread_list.append(random_face_imread)

        # Select the random face's bounding box
        random_face_bounds = grouped_bounding_boxes[i][indx_random_face]
        random_face_bounds_list.append(random_face_bounds)

    return random_faces, random_face_imread_list, random_face_bounds_list

def movies_to_analyse(title_1, title_2):
    faces_movie_1, imread_faces_movie_1, imread_faces_bounds_movie_1 = movie_to_analyse(title_1)
    faces_movie_2, imread_faces_movie_2, imread_faces_bounds_movie_2 = movie_to_analyse(title_2)

    return faces_movie_1, imread_faces_movie_1, imread_faces_bounds_movie_1, faces_movie_2, imread_faces_movie_2, imread_faces_bounds_movie_2
