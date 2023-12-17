
# label_parser_upwork


## Setup Instructions

### 1. Environment Setup
First, create a `.env` file in the root of the project and add the following AWS credentials and region information:

```
aws_access_key_id = 'add your key here' 
aws_secret_access_key = 'add your key here'
aws_region_name = 'add the region'
```

### 2. Install Requirements
Install the necessary dependencies by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

### 3. Usage

#### Generating Image Results
To generate images from a single PDF without OCR, use the following command:

```bash
python .\caption_parser.py --input_path .\input\Attention.pdf --output_folder .\outputs\ --no_ocr
```

#### Generating Captions Text Output (Requires OCR)
To generate captions text output from a single PDF using OCR, run:

```bash
python .\caption_parser.py --input_path .\input\Attention.pdf --output_folder .\outputs\
```

#### Running on Multiple PDFs
For processing multiple PDFs (assuming all PDFs are in the `input` folder), use:

```bash
python .\caption_parser.py --input_path .\input --output_folder .\outputs\ --no_ocr
```
