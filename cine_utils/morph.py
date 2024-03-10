# Import Statements
import numpy as np
from gradio_client import Client
import json

def image_mixer_api(path_1, path_2):
    """
    API that calls the image-mixer-demo app from HuggingFace
    """
    client = Client("https://lambdalabs-image-mixer-demo.hf.space/")

    result = client.predict(
				"Image",	# str  in 'Input 0 type' Radio component
				"Image",	# str  in 'Input 1 type' Radio component
				"Nothing",	# str  in 'Input 2 type' Radio component
				"Nothing",	# str  in 'Input 3 type' Radio component
				"Nothing",	# str  in 'Input 4 type' Radio component
				"",	# str  in 'Text or Image URL' Textbox component
				"",	# str  in 'Text or Image URL' Textbox component
				"",	# str  in 'Text or Image URL' Textbox component
				"",	# str  in 'Text or Image URL' Textbox component
				"",	# str  in 'Text or Image URL' Textbox component
				path_1,	# str (filepath or URL to image) in 'Image' Image component
				path_2,	# str (filepath or URL to image) in 'Image' Image component
				"",	# str (filepath or URL to image) in 'Image' Image component
				"",	# str (filepath or URL to image) in 'Image' Image component
				"",	# str (filepath or URL to image) in 'Image' Image component
				2.5,	# int | float (numeric value between 0 and 5) in 'Strength' Slider component
				2.5,	# int | float (numeric value between 0 and 5) in 'Strength' Slider component
				0,	# int | float (numeric value between 0 and 5) in 'Strength' Slider component
				0,	# int | float (numeric value between 0 and 5) in 'Strength' Slider component
				0,	# int | float (numeric value between 0 and 5) in 'Strength' Slider component
				3,	# int | float (numeric value between 1 and 10) in 'CFG scale' Slider component
				1,	# int | float (numeric value between 1 and 1) in 'Num samples' Slider component
				0,	# int | float (numeric value between 0 and 10000) in 'Seed' Slider component
				10,	# int | float (numeric value between 10 and 100) in 'Steps' Slider component
				fn_index=5
    )

    path_to_json = result + "/captions.json"

    # Open the JSON file and read its contents
    with open(path_to_json, "r") as file:
        data = json.load(file)

    return list(data.keys())[0]

def index_face(faces_title, selected_image):
    """
    Function that returns the index of the selected image in the faces_title list

    Returns:
        indx_image: Index of selected image in faces_title
    """
    # Flatten the list of faces arrays
    flattened_arrays = [arr.flatten() for arr in np.array(faces_title)]

    # Flatten the selected image array
    flattened_target = np.asarray(selected_image).flatten()

    # Find the index of the selected image array
    indx_image = next((i for i, arr in enumerate(flattened_arrays) if np.array_equal(arr, flattened_target)), None)

    return indx_image
