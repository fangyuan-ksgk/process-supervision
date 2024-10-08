from pdf2image import convert_from_path
import base64
from openai import OpenAI 
import os
import io 
import re 
import json
import glob
from pypdf import PdfReader
from .prompt import PARSE_RULE_PROMPT
from .rule import Rule
from PIL import Image
import ast
import subprocess


oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_oai_response(prompt, system_prompt="You are a helpful assistant", img=None, img_type="base64"):
    
    if isinstance(prompt, str):
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        msg = [
            {"role": "system", "content": system_prompt},
        ]
        msg.extend(prompt)
    
    if img is not None and img_type is not None:
        if isinstance(img, str):
            img = [img]
        image_content = []
        for _img in img:
            image_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img}"}})
            
        text = msg[-1]["content"]
        text_content = [{"type": "text", "text": text}]
        
        msg.append({
            "role": "user",
            "content": text_content + image_content,
        })
        
    response = oai_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=msg,
    )
    
    # print("Response: ", response.choices[0].message.content)
    
    return response.choices[0].message.content



def get_pdf_page_count(pdf_path):
    try:
        result = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Pages:'):
                return int(line.split(':')[1].strip())
    except subprocess.CalledProcessError:
        print("Error: pdfinfo command failed. Make sure it's installed and the PDF is valid.")
    except FileNotFoundError:
        print("Error: pdfinfo command not found. Make sure it's installed on your system.")
    return None


def get_pdf_contents(pdf_file, first_page=1, last_page=1, max_height=1000):
    # Convert the specified pages of the PDF to images
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)

    pdf_base64_images = []
    for pdf_image in images:
        # Calculate new dimensions while maintaining aspect ratio
        aspect_ratio = pdf_image.width / pdf_image.height
        new_height = min(pdf_image.height, max_height)
        new_width = int(new_height * aspect_ratio)
        
        # Resize the image
        pdf_image = pdf_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to grayscale
        pdf_image = pdf_image.convert('L')
        
        # Convert PIL Image to base64 string with reduced quality
        buffered = io.BytesIO()
        pdf_image.save(buffered, format="JPEG", quality=50)  # Reduced quality
        pdf_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        pdf_base64_images.append(pdf_image_base64)
        
    return pdf_base64_images


def chunk_to_rule_list(pdf_path, l, r):
    img = get_pdf_contents(pdf_path, first_page=l, last_page=r)
    prompt = PARSE_RULE_PROMPT
    response = get_oai_response(prompt, img=img)
    rule_list = extract_json_from_content(response)
    return rule_list





def pdf_to_img(pdf_file, first_page=1, last_page=1):
    # Convert the first page of the PDF to an image
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)        
    return images[0]

def get_pdf_text(pdf_path: str) -> str:
    pdf_text = ""
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    for i in range(number_of_pages):
        page = reader.pages[i]
        pdf_text += page.extract_text()
    return pdf_text


def extract_json_from_content(content):
    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        try:
            json_data = json.loads(match.group(1))
            return json_data
        except json.JSONDecodeError:
            print("Error: JSON parsing failed.")
            return []
    else:
        print("No JSON content found within ```json ... ``` blocks.")
        return []


def preprocess_rules(rule_dir: str = "database/rulebook"):
    pdf_paths = glob.glob(f"{rule_dir}/*.pdf")
    print("Preprocessing Rules...")
    rules = []
    for file_path in pdf_paths:
        print(f"Processing file: {file_path}")
        
        if file_path.endswith(".pdf"):
            rule_text = get_pdf_text(file_path)
            print(f"Extracted text from PDF: {file_path}")
        else:
            raise ValueError("Unknown file format")

        prompt = PARSE_RULE_PROMPT
        print("Generated prompt for rule parsing")
        max_tries = 3
        parsed_rules = None
        while parsed_rules is None and max_tries > 0:
            print(f"Attempt {4 - max_tries} to parse rule")
            response = get_oai_response(prompt)
            print("Received response from OpenAI")
            parsed_rules = extract_json_from_content(response)
            print("Parsed JSON response")
            if parsed_rules is None:
                max_tries -= 1
                continue
            
            for parsed_rule_dict in parsed_rules:
                try:
                    rule = Rule(**parsed_rule_dict)
                    rule.file_path = file_path
                    rule.save(rule_dir)
                    rules.append(rule)
                    print(f"Successfully parsed and saved rule: {file_path}")
                except Exception as e:
                    print(f"Error parsing rule: {e}")
        
        if parsed_rules is None:
            print(f"Failed to parse Rule after all attempts: {file_path}")
    
    return rules


def file_to_img(file_path):
    if file_path.endswith(".pdf"):
        img = pdf_to_img(file_path)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        with Image.open(file_path) as img_raw:
            buffered = io.BytesIO()
            img_raw.save(buffered, format="PNG")
            img = buffered.getvalue()
    else:
        raise ValueError("Unknown file format")
    return img