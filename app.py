from flask import Flask, request, redirect, url_for, send_file, render_template
import os
import pandas as pd
import pdfplumber
from PyPDF2 import PdfWriter, PdfReader
import zipfile
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import re
import pytesseract

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])

# Function to preprocess the image for better OCR accuracy
def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Apply a sharpen filter
    image = image.filter(ImageFilter.SHARPEN)
    return image

# Function to process an image and extract tracking number
def process_image(image):
    # Crop the image to the second half
    width, height = image.size
    cropped_image = image.crop((0, height // 2, width, height))
    # Preprocess the image
    preprocessed_image = preprocess_image(cropped_image)
    # Convert the image to text using pytesseract
    text = pytesseract.image_to_string(preprocessed_image)
    # Split the text into lines
    lines = text.split('\n')

    tracking_lines = []
    tracking_number_pattern = re.compile(r'\d{1,4}(\s*\d{1,4}){4,21}')
    for line in lines:
        if "TRACKING #:" in line:
            # Extract the number after "TRACKING #:" and remove whitespaces
            tracking_number = line.split("TRACKING #:")[-1].strip()
            tracking_number = re.sub(r'\s+', '', tracking_number)
            tracking_lines.append(tracking_number)
        else:
            match = tracking_number_pattern.search(line)
            if match:
                tracking_number = match.group().strip()
                tracking_number = re.sub(r'\s+', '', tracking_number)
                tracking_lines.append(tracking_number)

    if len(tracking_lines) == 0:
        print('not found!')
        tracking_lines.append(0)

    return tracking_lines

# Function to extract lines containing "TRACKING" or a 22-digit number from images in PDF and save them to a DataFrame
def extract_tracking_lines_from_pdf(pdf_path):
    data = []

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages):
            # Extract the image from the page
            image = pdf.pages[page_number].to_image(resolution=400).original
            # Process the image and extract tracking lines
            lines = process_image(image)
            if lines:
                for line in lines:
                    data.append({'Page Number': page_number + 1, 'Tracking Number': line})
                    print(f"Page {page_number+1}: {line}")

    df = pd.DataFrame(data)
    return df, num_pages

# Function to split the PDF into individual pages named by Order Number
from PyPDF2 import PdfWriter, PdfReader

# Function to split the PDF into individual pages named by Order Number
def split_pdf_by_order_number(pdf_path, df, output_folder):
    with open(pdf_path, 'rb') as infile:
        reader = PdfReader(infile)
        for index, row in df.iterrows():
            page_number = row['Page Number'] - 1  # PDF page index is 0-based
            order_number = row['Order Number']

            # Create a new PDF writer for the single page
            writer = PdfWriter()
            writer.add_page(reader.pages[page_number])

            # Define the output path
            output_path = os.path.join(output_folder, f'{order_number}.pdf')

            # Write the single page to a new PDF
            with open(output_path, 'wb') as out_pdf:
                writer.write(out_pdf)
            print(f'Saved: {output_path}')


# Route for the file upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle file uploads and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files or 'csv_file' not in request.files:
        return redirect(request.url)

    pdf_file = request.files['pdf_file']
    csv_file = request.files['csv_file']

    if pdf_file.filename == '' or csv_file.filename == '':
        return redirect(request.url)

    if pdf_file and csv_file:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)

        pdf_file.save(pdf_path)
        csv_file.save(csv_path)

        extracted_df, num_pages = extract_tracking_lines_from_pdf(pdf_path)
        csv_df = pd.read_csv(csv_path)
        csv_df.rename(columns={'tracking_number': 'Tracking Number', 'number': 'Order Number'}, inplace=True)

        # Merge the extracted DataFrame with the CSV DataFrame
        merged_df = pd.merge(extracted_df, csv_df, on='Tracking Number', how='left')
        if merged_df['Order Number'].isnull().any():
            missing_tracking_numbers = merged_df[merged_df['Order Number'].isnull()]['Tracking Number']
            print(f"Warning: The following tracking numbers were not found in the CSV: {missing_tracking_numbers.tolist()}")
            merged_df['Order Number'].fillna('NOT_FOUND', inplace=True)
        split_pdf_by_order_number(pdf_path, merged_df, app.config['OUTPUT_FOLDER'])

        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(app.config['OUTPUT_FOLDER']):
                for file in files:
                    if file.endswith('.pdf'):
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), app.config['OUTPUT_FOLDER']))

        return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
