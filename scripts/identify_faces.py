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
from dotenv import load_dotenv
import random

# Load the credentials
load_dotenv()

# Get the path to the key file from the environment variable
key_file_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Authenticate with Google Cloud using the key file
client = storage.Client.from_service_account_json(key_file_path)

bucket_name = "cine_ethics"
bucket = client.get_bucket(bucket_name)

synopsis_screengrabs_df = pd.read_csv("raw_data/synopsis_screengrabs_final_table.csv")

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
    sample_size = 100 # Number of images to sample as opposed to the full 1k
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
            #print("Processing", blob.name, "...")

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
    path_encodings = os.path.join("raw_data/deep_face_encodings/", str(movie_info["title"].values))
    path_encodings = str(path_encodings)

    #print("Creating directory", path_encodings)

    # Ensure that the directory exists
    os.makedirs(str(path_encodings), exist_ok=True)

    # Dump the facial encodings data to disk
    file_name = movie_info["title"].values + ".pickle"
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
    numUniqueFaces = len(np.where(labelIDs > -1)[0])

    grouped_faces = []

    # Loop over the unique face integers
    for labelID in labelIDs:
        # Find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes from the set
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

        # initialize the list of faces to include in the montage
        faces = []

        # loop over the sampled indexes
        for i in idxs:
            # Load the input image from Google Cloud Storage and extract the face ROI
            blob = client.bucket(bucket_name).blob(data[i]["imagePath"])
            image_bytes = blob.download_as_string()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            (top, right, bottom, left) = data[i]["loc"]
            face = image[top:bottom, left:right]

            # force resize the face ROI to 96x96 and then add it to the
            # faces montage list
            # face = cv2.resize(face, (96, 96))
            faces.append(face)

        grouped_faces.append(faces)

    # List to store one random photo for each character
    random_faces = []
    # Return one of each in grouped faces
    for i in range(len(grouped_faces)):
        random_face = random.sample(grouped_faces[i], 1)
        random_faces.append(random_face)

    return random_faces

def movies_to_analyse(title_1, title_2):
    faces_movie_1 = movie_to_analyse(title_1)
    faces_movie_2 = movie_to_analyse(title_2)

    return faces_movie_1, faces_movie_2
