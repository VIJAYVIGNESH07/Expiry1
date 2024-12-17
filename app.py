import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import easyocr
import json
import re
from datetime import datetime
import pandas as pd
import os


# Load models and resources
yolo_model = YOLO('best1143images(100).pt')  # YOLO model for expiry detection
expiry_keras_model = load_model('date_detection_model1.keras')  # Keras model for expiry detection
brand_keras_model = load_model('my_model.keras')  # Keras model for brand detection

# Load class indices for brand detection
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices mapping for decoding
class_indices_reversed = {v: k for k, v in class_indices.items()}

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess the image for Keras model
def preprocess_for_keras(image):
    image = cv2.resize(image, (224, 224))  # Resize to MobileNetV2 input size
    image = image.astype("float32") / 255.0  # Normalize pixel values
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to detect the brand
def detect_brand(image):
    preprocessed_image = preprocess_for_keras(image)
    predictions = brand_keras_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    brand_name = class_indices_reversed.get(predicted_class, "Unknown")
    return brand_name

# Function to extract dates from text using regex
def extract_dates(text):
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',    # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{4}/\d{2}/\d{2}\b',    # YYYY/MM/DD
        r'\b\d{2}\.\d{2}\.\d{4}\b',  # MM.DD.YYYY or DD.MM.YYYY
        r'\b\d{4}\.\d{2}\.\d{2}\b',  # YYYY.MM.DD
        r'\b\d{2}-\d{2}-\d{4}\b',    # MM-DD-YYYY or DD-MM-YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',    # YYYY-MM-DD
    ]
    detected_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                detected_dates.append(parse_date(match))
            except ValueError:
                continue
    return detected_dates

# Function to parse date string to datetime object
def parse_date(date_str):
    date_formats = [
        "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%m.%d.%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%m-%d-%Y", "%d-%m-%Y", "%Y-%m-%d",
    ]
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    raise ValueError(f"Date format not recognized: {date_str}")

# Function to select the latest date
def get_latest_date(dates):
    if not dates:
        return None
    return max(dates)

# Function to compare dates and determine expiration status
def compare_dates(latest_date):
    current_date = datetime.now()

    if latest_date < current_date:
        expired_status = "Yes"
        status_message = f"Expired on {latest_date.strftime('%d-%m-%Y')}"
        life_span = "NA"
    else:
        expired_status = "No"
        status_message = f"Valid until {latest_date.strftime('%d-%m-%Y')}"
        life_span = (latest_date - current_date).days

    return expired_status, life_span, status_message

# Function to save details to the Excel file
def save_details_to_excel(time, brand_name, expiry_date, expired_status, life_span, output_file):
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
    else:
        df = pd.DataFrame(columns=['Sno', 'Time', 'BrandName', 'Expiry Date', 'Expired Status', 'Life Span'])

    # Add new entry to the DataFrame
    new_row = pd.DataFrame({
        'Sno': [len(df) + 1],
        'Time': [time],
        'BrandName': [brand_name],
        'Expiry Date': [expiry_date.strftime('%d-%m-%Y') if expiry_date else "NA"],
        'Expired Status': [expired_status],
        'Life Span': [life_span if life_span != "NA" else "NA"]
    })
    df = pd.concat([df, new_row], ignore_index=True)

    # Save to the Excel file
    df.to_excel(output_file, index=False)

# Streamlit Frontend
st.set_page_config(page_title="Expiry Date Detection", layout="centered")
st.title("Expiry Date Detection")
st.write('"Leverage AI to detect expiration dates quickly and accurately."')

# File upload sections
brand_image_file = st.file_uploader("Upload Brand Image", type=["jpg", "png", "jpeg"], key="brand")
expiry_image_file = st.file_uploader("Upload Expiry Date Image", type=["jpg", "png", "jpeg"], key="expiry")

output_excel_file = 'Expiry_Brand_Details3.xlsx'

if st.button("Submit for Expiry Detection"):
    if brand_image_file and expiry_image_file:
        # Load the images
        brand_image = cv2.imdecode(np.frombuffer(brand_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        expiry_image = cv2.imdecode(np.frombuffer(expiry_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect brand
        brand_name = detect_brand(brand_image)

        # Detect expiry date
        results = yolo_model(expiry_image)
        detected_dates = []
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                detected_region = expiry_image[y1:y2, x1:x2]

                preprocessed_region = preprocess_for_keras(detected_region)
                prediction = expiry_keras_model.predict(preprocessed_region)[0][0]

                if prediction < 0.5:
                    ocr_results = reader.readtext(detected_region)
                    for _, text, _ in ocr_results:
                        detected_dates.extend(extract_dates(text))

        # Select the latest date
        latest_date = get_latest_date(detected_dates)

        # Determine expiration status and life span
        expired_status, life_span, status_message = compare_dates(latest_date)

        # Save details to Excel
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_details_to_excel(current_time, brand_name, latest_date, expired_status, life_span, output_excel_file)

        # Display results
        st.success("Detection Complete!")
        st.write(f"**Brand Name:** {brand_name}")
        st.write(f"**Expiry Date:** {latest_date.strftime('%d-%m-%Y') if latest_date else 'NA'}")
        st.write(f"**Status:** {status_message}")

        with open(output_excel_file, "rb") as excel_file:
            st.download_button("Download Excel", excel_file, "Expiry_Brand_Details.xlsx")
    else:
        st.error("Please upload both images.")

