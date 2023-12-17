import boto3
from config import settings
from config import settings
from pdf2image import convert_from_bytes
from io import BytesIO

def get_aws_ocr_text(img_bytes, aws_access_key_id=settings.aws_access_key_id, aws_secret_access_key=settings.aws_secret_access_key, region_name=settings.aws_region_name):
    # Initialize the Textract client
    client = boto3.client('textract', region_name=region_name, aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)

    # Call Textract to analyze the document image
    response = client.detect_document_text(Document={'Bytes': img_bytes})
    lines = []

    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            lines.append(item["Text"])

    # Join the lines into a single string separated by newlines
    return '\n'.join(lines)


def pdf_to_images(uploaded_file, dpi=100):
    # Convert the uploaded file to bytes
    pdf_bytes = uploaded_file.read()

    # Convert PDF bytes to a list of images
    images = convert_from_bytes(pdf_bytes, dpi=dpi)

    # Convert images to black and white
    bw_images = []
    for image in images:
        # Convert to grayscale
        grayscale = image.convert('L')
        # Convert to black and white
        bw = grayscale.point(lambda x: 0 if x < 128 else 255, '1')
        bw_images.append(bw)

    return bw_images