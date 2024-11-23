# Bank Statement OCR Converter

A web application that automatically converts bank statement images into structured Excel files using advanced OCR technology.

## Features

- Upload bank statement images (PNG, JPG, JPEG, TIFF, BMP)
- Advanced image preprocessing for better OCR accuracy
- Extracts account information and transactions
- Generates formatted Excel files with original filename
- Modern, responsive web interface
- Real-time preview of extracted data

## Requirements

- Python 3.12
- Tesseract OCR v5.3.3
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd BankProject
```

2. Install Tesseract OCR:
- Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Create uploads directory:
```bash
mkdir uploads
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`
3. Upload a bank statement image
4. Preview the extracted data
5. Download the formatted Excel file

## Data Extraction

The application extracts:
- Bank Name
- Account Name
- Account Number
- Statement Date
- Transactions (Date, Description, Withdrawal, Deposit, Balance)

## Excel Output

- Two sheets: Account Info and Transactions
- Formatted headers and styling
- Currency formatting for amounts
- Auto-adjusted column widths
- Original filename preserved

## Security

- File size limit: 16MB
- Secure filename handling
- Temporary file cleanup
- Input validation

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details
