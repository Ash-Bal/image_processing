import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
import random
import torch
from PIL import Image
import numpy as np
import boto3
from boto3.dynamodb.types import TypeDeserializer
from dynamodb_connection import DynamoDBConnection
from st_files_connection import FilesConnection
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Cat or Dog?", page_icon=":dog:", initial_sidebar_state="auto"
)

hide_streamlit_style = """ 
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    st.image("cosmo.jpeg")
    st.title("Cat or Dog?")
    st.subheader(
        "This ViT has been fine-tuned on a dataset of cats and dogs. If you aren't able to distinguish between cat's and dogs, this model can help!"
    )


@st.cache(allow_output_mutation=True)
def initialize_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model.load_state_dict(torch.load("cat_v_dog.pth"))
    model.eval()
    return model, feature_extractor


# def dynamo_to_python(dynamo_object: dict) -> dict:
#     deserializer = TypeDeserializer()
#     return {k: deserializer.deserialize(v) for k, v in dynamo_object.scan()["Items"]}


def dynamo_to_python(dynamo_items):
    deserializer = TypeDeserializer()
    data = {"Submission": {}, "File": {}, "Result": {}}

    for item in dynamo_items:
        data["Submission"][str(item["submission_num"])] = item["submission_num"]
        data["File"][str(item["submission_num"])] = item["file_name"]
        data["Result"][str(item["submission_num"])] = item["result"]

    return data


dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("Submissions")
data_dict = table.scan()


with st.spinner("Model is being loaded.."):
    model, feature_extractor = initialize_model()

st.write(
    """
         # Cat or Dog?
         """
)

file = st.file_uploader("", type=["jpg", "png"])


def import_and_predict(image_data, model, feature_extractor):
    img = feature_extractor(image_data, return_tensors="pt")
    output = model(**img)
    animal_class = output.logits.softmax(1).argmax(1)
    label = "cat" if animal_class == 0 else "dog"
    return label


if file is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model, feature_extractor)
    # x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    # st.sidebar.error("Accuracy : " + str(x) + " %")

    string = "Detected Animal : " + predictions
    st.sidebar.warning(string)

    submission_num = data_dict["Count"]
    fname = file.name
    res = predictions

    table.put_item(
        Item={
            "submission_num": submission_num,
            "file_name": fname,
            "result": predictions,
        }
    )
    # table = dynamodb.Table("Submissions")
    data_dict = table.scan()

if data_dict["Count"] == 0:
    st.text("There are no previous submissions!")
else:
    python_json = dynamo_to_python(data_dict["Items"])
    df = pd.DataFrame.from_dict(python_json)
    st.write(df)
