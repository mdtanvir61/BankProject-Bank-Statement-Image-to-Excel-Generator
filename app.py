from flask import Flask, request, render_template, send_file, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import logging
import json
from pathlib import Path
from google.cloud import vision
import io
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# Initialize Google Cloud Vision client
vision_client = vision.ImageAnnotatorClient()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_bytes):
    """Preprocess the image for better OCR results."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Convert back to bytes
    success, buffer = cv2.imencode('.png', denoised)
    return buffer.tobytes() if success else image_bytes

def extract_text(image_bytes):
    """Extract text using Google Cloud Vision API."""
    try:
        # Preprocess image
        processed_bytes = preprocess_image(image_bytes)
        
        # Create image object
        image = vision.Image(content=processed_bytes)
        
        # Perform OCR
        response = vision_client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Error: {response.error.message}")
            
        return response.full_text_annotation.text
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise

def parse_statement(text):
    """Parse the extracted text into structured data."""
    lines = text.split('\n')
    
    result = {
        'account_info': {
            'bank_name': '',
            'account_name': '',
            'account_number': '',
            'statement_date': ''
        },
        'transactions': []
    }
    
    current_section = None
    for line in lines:
        line = line.strip()
        
        # Extract bank name
        if 'BANK' in line.upper():
            result['account_info']['bank_name'] = line
            
        # Extract account number
        elif 'ACCOUNT' in line.upper() and 'NO' in line.upper():
            numbers = ''.join(filter(str.isdigit, line))
            if numbers:
                result['account_info']['account_number'] = numbers
                
        # Extract date
        elif 'DATE' in line.upper():
            import re
            date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
            dates = re.findall(date_pattern, line)
            if dates:
                result['account_info']['statement_date'] = dates[0]
                
        # Extract transactions
        elif any(word in line.upper() for word in ['CREDIT', 'DEBIT', 'TRANSFER', 'DEPOSIT', 'WITHDRAWAL']):
            try:
                # Parse transaction line
                import re
                amounts = re.findall(r'\d+,?\d*\.?\d*', line)
                if amounts:
                    transaction = {
                        'date': '',
                        'description': line,
                        'withdrawal': amounts[-2] if len(amounts) >= 2 else '0',
                        'deposit': amounts[-1] if amounts else '0',
                        'balance': amounts[0] if amounts else '0'
                    }
                    result['transactions'].append(transaction)
            except Exception as e:
                logger.warning(f"Failed to parse transaction line: {line}")
                
    return result

def create_excel(data, filename):
    """Create Excel file from extracted data."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            excel_path = tmp.name
            
        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write account info
            account_df = pd.DataFrame([data['account_info']])
            account_df.to_excel(writer, sheet_name='Account Info', index=False)
            
            # Write transactions
            if data['transactions']:
                trans_df = pd.DataFrame(data['transactions'])
                trans_df.to_excel(writer, sheet_name='Transactions', index=False)
                
                # Auto-adjust columns width
                worksheet = writer.sheets['Transactions']
                for idx, col in enumerate(trans_df.columns):
                    max_length = max(
                        trans_df[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        return excel_path
        
    except Exception as e:
        logger.error(f"Error creating Excel file: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Read file bytes
        file_bytes = file.read()
        
        try:
            # Extract and process text
            extracted_text = extract_text(file_bytes)
            result = parse_statement(extracted_text)
            
            # Create Excel file
            excel_path = create_excel(result, file.filename)
            
            # Return the Excel file
            return send_file(
                excel_path,
                as_attachment=True,
                download_name=f"{Path(file.filename).stem}.xlsx",
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        finally:
            # Cleanup temporary Excel file
            if 'excel_path' in locals() and os.path.exists(excel_path):
                os.remove(excel_path)
                
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)
