
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import math
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction,get_prediction
st.set_page_config(page_title="PCB Component Detection")
import time

# CSS styles
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 30px;
        margin-bottom: 30px;
        text-decoration: underline;
    }
    .upload {
        border: 1px dashed #ddd;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .btn-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .btn-container .btn {
        margin: 0 10px;
    }
    .component-table {
        margin-top: 30px;
        text-decoration: underline;
    }
    .legend {
        margin-top: 30px;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Loading Models for Capacitor&Resistor and IC
model_cap=YOLO("Weights/best_cap.pt")
model_cap.overrides['classes']=[0]

model_ic=YOLO("Weights/best_ic.pt")
model_ic.overrides['classes']=[1]

model_res=YOLO("Weights/best_res.pt")
model_res.overrides['classes']=[2]



# model_fiducial=YOLO(r"")

def label_override(result):
    label_mapping = {
        'IC': 'IC',
        'Capacitor': 'C',
        'Resistor': 'R'
    }

    for prediction in result.object_prediction_list:
        label = prediction.category.name
        custom_label = label_mapping.get(label, label)
        prediction.category.name = custom_label

def detect(image, conf,filter):

    if filter=='Capacitor':
        model=model_cap
    elif filter=='Resistor':
        model=model_res
    elif filter=='IC':
        model=model_ic
    elif filter=='All':
        model=[model_ic,model_cap,model_res]


    if filter!='All':

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model=model,
            confidence_threshold=conf,
            device="cuda:0",
        )
        start_time = time.time()
        result = get_sliced_prediction(image, detection_model)
        inference_time = time.time() - start_time

        label_override(result)
        return result,inference_time
    
    else:

        detection_model_ic = AutoDetectionModel.from_pretrained(model_type='yolov8',model=model_ic, confidence_threshold=conf, device="cuda:0",)
        detection_model_res = AutoDetectionModel.from_pretrained(model_type='yolov8',model=model_res, confidence_threshold=conf, device="cuda:0",)
        detection_model_cap = AutoDetectionModel.from_pretrained(model_type='yolov8',model=model_cap, confidence_threshold=conf, device="cuda:0",)

        start_time = time.time()

        result_ic=get_sliced_prediction(image,detection_model_ic)
        result_cap=get_sliced_prediction(image,detection_model_cap)
        result_res=get_sliced_prediction(image,detection_model_res)

        inference_time = time.time() - start_time


    
        combined_results=result_ic
        combined_results.object_prediction_list.extend(result_cap.object_prediction_list)
        combined_results.object_prediction_list.extend(result_res.object_prediction_list)

        label_override(combined_results)

        return combined_results,inference_time


def quantity(results):
    dict = {'IC': 0, 'C': 0, 'R': 0}
    for i in results.object_prediction_list:
        dict[str(i.category.name)] += 1
    return dict["IC"], dict["C"], dict["R"]


def main():
    st.markdown("<h1 class='title'>PCB COMPONENT DETECTION</h1>", unsafe_allow_html=True)
    st.write("Upload an image and detect PCB components.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, channels="RGB", caption="Original Image")
        col1,col2=st.columns(2)

        conf = st.slider(label='Confidence Threshold', min_value=0, max_value=100, value=50, step=5)
        conf = conf / 100.00
        with col1:
            option = st.selectbox("Hide Labels:", ("Yes", "No"), index=1)
            if option=='Yes':
                option=True
            else:
                option=False
        with col2:
            filter=st.selectbox("Filters :",("All","IC","Capacitor","Resistor"),index=0)

        btn_detect = st.button("Detect Objects")

        if btn_detect:
            with st.spinner("Detecting..."):
                results ,inf= detect(image, conf,filter)

                results.export_visuals(export_dir="./exports", hide_labels=option)
            
            st.success(f"Detection completed in {round(inf,3)} Seconds.")

            st.image("exports/prediction_visual.png", channels="RGB", caption="Inference Image")
            st.markdown("<h2 class='legend'>Legend</h2>", unsafe_allow_html=True)
            st.text("\t \tIC: Integrated Circuit   C:  Capacitor     R: Resistor  ")

            ic, cap, res = quantity(results)

            
            if filter=='IC':
                table = {"Components": ['IC'], "Quantity": [ic]}
            elif filter=='Capacitor':
                table = {"Components": ['Capacitor'], "Quantity": [cap]}
            elif filter=='Resistor':
                table = {"Components": ['Resistor'], "Quantity": [res]}
            else:
                table = {"Components": ['IC', 'Capacitor', 'Resistor'], "Quantity": [ic, cap, res]}

            df = pd.DataFrame(table)
            st.markdown("<h2 class='component-table'>Component Quantities</h2>", unsafe_allow_html=True)
            st.table(df)
            df=df.to_csv(index=False)
            st.download_button(
            "Download",
            df,
            "file.csv",
            "text/csv",
            key='download-csv' 
            )



if __name__ == "__main__":
    main()
