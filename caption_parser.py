
import fitz  # PyMuPDF
import numpy as np
import cv2
import layoutparser as lp
import cv2
import math
import pytesseract
import os
import argparse
from do_ocr import get_aws_ocr_text
from PyPDF2 import PdfReader
import re
import warnings
import json
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar
import fitz  # PyMuPDF
import re
from PyPDF2 import PdfReader
import layoutparser as lp
import cv2
import os
from pdf2image import convert_from_path
from PIL import Image
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

class TextBlock:
    def __init__(self, block, type):
        self.block = block
        self.type = type

class Rectangle:
    def __init__(self, x_1, y_1, x_2, y_2):
        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2

    def center(self):
        return ((self.x_1 + self.x_2) / 2, (self.y_1 + self.y_2) / 2)


class Layout:
    def __init__(self, _blocks, page_data={}):
        self._blocks = _blocks
        self.page_data = page_data

    def __str__(self):
        blocks_str = ', '.join(str(block) for block in self._blocks)
        return f"Layout(_blocks=[{blocks_str}], page_data={self.page_data})"





def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_closest_text_blocks(blocks):
    figure_blocks = [block for block in blocks if block.type == "Figure"]
    text_blocks = [block for block in blocks if block.type == "Text"]

    closest_blocks = []
    for figure in figure_blocks:
        figure_center = figure.block.center()
        min_distance = float('inf')
        closest_block = None

        for text in text_blocks:
            text_center = text.block.center()
            dist = distance(figure_center, text_center)
            if dist < min_distance:
                min_distance = dist
                closest_block = text

        closest_blocks.append(closest_block)

    return closest_blocks

def identify_closest_text_block(closest_blocks, original_blocks):
    closest_block_layout = []

    for closest in closest_blocks:
        if closest:
            closest_coords = (closest.block.x_1, closest.block.y_1, closest.block.x_2, closest.block.y_2)

            # Find and return the matching TextBlock from the original list
            for original in original_blocks:
                original_coords = (original.block.x_1, original.block.y_1, original.block.x_2, original.block.y_2)
                if closest_coords == original_coords:
                    closest_block_layout.append(original)

    return lp.Layout(closest_block_layout)


def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        images.append(img_bgr)
    return images

def process_page(image, model):
    layout = model.detect(image)
    layout_blocks = [TextBlock(Rectangle(b.block.x_1, b.block.y_1, b.block.x_2, b.block.y_2), b.type) for b in layout]
    return layout,layout_blocks

def create_output_dict(pdf_path, model):
    images = pdf_to_images(pdf_path)
    output_dict = {}

    for page_num, image in enumerate(images, start=1):
        layout, layout_blocks = process_page(image, model)
        figure_blocks = [b for b in layout_blocks if b.type == 'Figure']
        table_blocks = [b for b in layout_blocks if b.type == 'Table']
        page_dict = {'Figure': {}, 'Table': {}}

        for fig in figure_blocks:
            closest_blocks = find_closest_text_blocks([fig] + layout_blocks)
            closest_text = identify_closest_text_block(closest_blocks, layout)
            page_dict['Figure'][fig] = {'layout': lp.Layout([fig]), 'closest_text': closest_text}

        for table in table_blocks:
            closest_blocks = find_closest_text_blocks([table] + layout_blocks)
            closest_text = identify_closest_text_block(closest_blocks, layout)
            page_dict['Table'][table] = {'layout': table, 'closest_text': closest_text}

        output_dict[f'Page_{page_num}'] = page_dict

    return output_dict

def create_output_dict_crop_images_no_ocr(pdf_path, model, output_folder):
    pdf_base_name = os.path.basename(pdf_path).split('.')[0]
    images_save_path = os.path.join(output_folder, pdf_base_name)

    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)

    doc = fitz.open(pdf_path)
    images = pdf_to_images(pdf_path)
    output_dict = {}

    for page_num, image in enumerate(images, start=1):
        layout, layout_blocks = process_page(image, model)
        page = doc.load_page(page_num - 1)
        page_dict = {'Figure': {}}

        figure_count = 1
        for fig in layout_blocks:
            if fig.type == 'Figure':
                closest_blocks = find_closest_text_blocks([fig] + layout_blocks)
                closest_text = identify_closest_text_block(closest_blocks, layout_blocks)

                figure_name = f"Figure{figure_count}"
                fig_crop_path = os.path.join(images_save_path, f"Page{page_num}_{figure_name}.png")
                fig_crop = page.get_pixmap(clip=fitz.Rect(fig.block.x_1, fig.block.y_1, fig.block.x_2, fig.block.y_2))
                fig_crop.save(fig_crop_path)

                caption_text = ""
                if closest_text:
                    text_block = closest_text._blocks[0]
                    text_crop_path = os.path.join(images_save_path, f"Page{page_num}_{figure_name}_Text.png")
                    text_crop = page.get_pixmap(clip=fitz.Rect(text_block.block.x_1, text_block.block.y_1, text_block.block.x_2, text_block.block.y_2))
                    text_crop.save(text_crop_path)
                    #caption_text = pytesseract.image_to_string(text_crop_path)

                page_dict['Figure'][figure_name] = {'caption': caption_text}
                figure_count += 1

        output_dict[f'Page_{page_num}'] = page_dict

    doc.close()
    #return output_dict

def create_output_dict_crop_images_and_ocr(pdf_path, model, output_folder):
    pdf_base_name = os.path.basename(pdf_path).split('.')[0]
    images_save_path = os.path.join(output_folder, pdf_base_name)
    reader = PdfReader(pdf_path)

    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)

    doc = fitz.open(pdf_path)
    images = pdf_to_images(pdf_path)
    output_dict = {}

    # Extract all text from the PDF and split it into lines
    all_text = []
    line_to_page = {}  # Dictionary to map line ids to page numbers
    line_id = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            lines = text.split("\n")
            for line in lines:
                line_to_page[line_id] = page_num
                line_id += 1
                all_text.append(line.strip())
    lines_mention = '\n'.join(all_text).split('.')

    for page_num, image in enumerate(images, start=1):
        layout, layout_blocks = process_page(image, model)
        page = doc.load_page(page_num - 1)
        page_dict = {'Figure': {}}

        figure_count = 1
        for fig in layout_blocks:
            if fig.type == 'Figure':
                closest_blocks = find_closest_text_blocks([fig] + layout_blocks)
                closest_text = identify_closest_text_block(closest_blocks, layout_blocks)

                figure_name = f"Figure{figure_count}"
                fig_crop_path = os.path.join(images_save_path, f"Page{page_num}_{figure_name}.png")
                fig_crop = page.get_pixmap(clip=fitz.Rect(fig.block.x_1, fig.block.y_1, fig.block.x_2, fig.block.y_2))
                fig_crop.save(fig_crop_path)

                caption_text = ""
                if closest_text:
                    text_block = closest_text._blocks[0]
                    text_crop_path = os.path.join(images_save_path, f"Page{page_num}_{figure_name}_Text.png")
                    text_crop = page.get_pixmap(clip=fitz.Rect(text_block.block.x_1, text_block.block.y_1, text_block.block.x_2, text_block.block.y_2))
                    text_crop.save(text_crop_path)
                    img_bytes = text_crop.tobytes("png")
                    caption_text = get_aws_ocr_text(img_bytes)

                # Extract figure number from caption
                match = re.search(r'Figure\s?(\d+)', caption_text)
                
                if match:
                    figure_number = match.group(1)
                    mentions = []
                    for i, line in enumerate(lines_mention):
                        if f"Figure {figure_number}" in line:
                            
                            prev_line = lines_mention[i - 1] if i > 0 else ""
                            next_line = lines_mention[i + 1] if i < len(lines_mention) - 1 else ""
                            mentions.append({'prev_line': prev_line, 'mention_line': line, 'next_line': next_line})


                page_dict['Figure'][figure_name] = {'caption': caption_text, 'mentions': mentions}
                figure_count += 1

        output_dict[f'Page_{page_num}'] = page_dict

    doc.close()
    print(output_dict)
    return output_dict




def convert_image_bbox_to_pdf(image_bbox, dpi, image_height, buffer = 3):

    x0_img, y0_img, x1_img, y1_img = image_bbox
    scale_factor = 72 / dpi 
    x0_pdf = x0_img * scale_factor - buffer
    x1_pdf = x1_img * scale_factor + buffer

    y0_pdf = image_height - y1_img
    y1_pdf = image_height - y0_img
    y0_pdf, y1_pdf = [y * scale_factor for y in (y0_pdf, y1_pdf)]

    return (x0_pdf, y0_pdf, x1_pdf, y1_pdf)



def get_text_in_bbox(pdf_path, bbox, page_number, spacing='dynamic', include_new_line=False):
    x0, y0, x1, y1 = bbox
    text_within_bbox = ""
    last_char = None

    for page_layout in extract_pages(pdf_path, page_numbers=[page_number-1]):  # Adjusting for zero-based index
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        for char in line:
                            if isinstance(char, LTChar):
                                char_bbox = char.bbox
                                if (char_bbox[0] >= x0 and char_bbox[2] <= x1 and
                                    char_bbox[1] >= y0 and char_bbox[3] <= y1):
                                    if spacing == 'dynamic' and last_char:
                                        gap = char_bbox[0] - last_char[2]
                                        avg_char_width = (last_char[2] - last_char[0] + char_bbox[2] - char_bbox[0]) / 2
                                        if gap > avg_char_width * 0.3:  
                                            text_within_bbox += " "
                                    elif spacing == 'fixed' and last_char:
                                        if (char_bbox[0] - last_char[2]) > (char_bbox[2] - char_bbox[0]) * 0.5:
                                            text_within_bbox += " "
                                    text_within_bbox += char.get_text()
                                    last_char = char_bbox
                        if include_new_line:
                            text_within_bbox += "\n"
    return text_within_bbox



def convert_pdf_to_images(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return images  # List of PIL Image objects
    
def convert_pdf_to_images(pdf_path, dpi=300):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Process each image
    processed_images = []
    for img in images:
        # Convert PIL image to NumPy array
        img_np = np.array(img)

        # Convert RGB (default in PIL) to BGR (used in OpenCV)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        processed_images.append(img_bgr)

    return processed_images


def create_output_native_pdf(pdf_path, model, output_folder, dpi=300):
    pdf_base_name = os.path.basename(pdf_path).split('.')[0]
    images_save_path = os.path.join(output_folder, pdf_base_name)

    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)

    doc = fitz.open(pdf_path)
    images = convert_pdf_to_images(pdf_path)
    output_dict = {}
    all_text = []
    line_to_page = {}  # Dictionary to map line ids to page numbers
    line_id = 0
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
 
        if text:
            lines = text.split("\n")
            for line in lines:
                line_to_page[line_id] = page_num
                line_id += 1
                all_text.append(line.strip())

    
    #lines_mention = '\n'.join(all_text).split('.')
                
    lines_mention =  re.split(r'(?<!\d)\.(?=\s)|(?<=\d)\.(?=\s)', '\n'.join(all_text))

    for page_num, image in enumerate(images, start=1):
        layout, layout_blocks = process_page(image, model)
        page = doc.load_page(page_num - 1)
        page_dict = {'Figure': {}}

        figure_count = 1
        for fig in layout_blocks:
            if fig.type == 'Figure':
                closest_blocks = find_closest_text_blocks([fig] + layout_blocks)
                closest_text = identify_closest_text_block(closest_blocks, layout_blocks)

                figure_name = f"Figure{figure_count}"
                fig_crop_path = os.path.join(images_save_path, f"Page{page_num}_{figure_name}.png")
                #fig_crop = page.get_pixmap(clip=fitz.Rect(fig.block.x_1, fig.block.y_1, fig.block.x_2, fig.block.y_2))
                #fig_crop.save(fig_crop_path)
                x1, y1, x2, y2 = int(fig.block.x_1), int(fig.block.y_1), int(fig.block.x_2), int(fig.block.y_2)
                fig_crop = image[y1:y2, x1:x2]
                cv2.imwrite(fig_crop_path, fig_crop)

                caption_text = ""
                if closest_text:
                    text_block = closest_text._blocks[0]
                    pdf_bbox = convert_image_bbox_to_pdf((text_block.block.x_1, text_block.block.y_1, text_block.block.x_2, text_block.block.y_2), dpi, image.shape[0])
                    caption_text = get_text_in_bbox(pdf_path, pdf_bbox, page_num, spacing='dynamic', include_new_line=False)
                    print(caption_text)

                # Extract figure number from caption
                match = re.search(r'Figure\s?(\d+)', caption_text)
                mentions = []
                if match:
                    figure_number = match.group(1)
                    
                    for i, line in enumerate(lines_mention):
                        if f"Figure {figure_number}" in line:
                            
                            prev_line = lines_mention[i - 1] if i > 0 else ""
                            next_line = lines_mention[i + 1] if i < len(lines_mention) - 1 else ""
                            mentions.append({'prev_line': prev_line, 'mention_line': line, 'next_line': next_line})


                page_dict['Figure'][figure_name] = {'caption': caption_text, 'mentions': mentions}
                figure_count += 1

        output_dict[f'Page_{page_num}'] = page_dict

    doc.close()
    #print(output_dict)
    return output_dict


def process_pdf(pdf_path, model, output_folder, do_ocr=False):

    if do_ocr:
        return create_output_native_pdf(pdf_path, model, output_folder)
    else:
        create_output_dict_crop_images_no_ocr(pdf_path, model, output_folder)

def save_dict_to_json(output_dict, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_dict, file, ensure_ascii=False, indent=4)

def main(input_path, output_folder, do_ocr ,model):
    save_ocr = True if do_ocr else False
    if os.path.isfile(input_path) and input_path.endswith('.pdf'):
        out = process_pdf(input_path, model, output_folder, do_ocr)
        if save_ocr:
            json_file_path = os.path.join(output_folder, os.path.basename(input_path).split('.')[0] + '_captions.json')
            save_dict_to_json(out, json_file_path)

    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(input_path, file)
                
                out = process_pdf(pdf_path, model, output_folder, do_ocr)
                if save_ocr:
                    json_file_path = os.path.join(output_folder, os.path.basename(pdf_path).split('.')[0] + '_captions.json')
                    save_dict_to_json(out, json_file_path)
    else:
        print(f"The input path {input_path} is not valid. Please specify a PDF file or a directory containing PDF files.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process a PDF or a directory of PDFs.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to a PDF file or a directory containing PDF files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder where results will be saved')
    parser.add_argument('--no_ocr', action='store_true', help='Flag to disable OCR')
    
    args = parser.parse_args()

    # Initialize your layout parser model here
    # model = lp.Detectron2LayoutModel(config_path='path_to_config.yml', label_map={0: 'Text', 1: 'Title', ...})
    model = lp.Detectron2LayoutModel(config_path = 'config.yml', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

    do_ocr = not args.no_ocr 

    print(f"extract text: {do_ocr}")
    main(args.input_path, args.output_folder, do_ocr,model)
