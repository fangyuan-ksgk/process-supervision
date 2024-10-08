from pdf2image import convert_from_path
import base64
from openai import OpenAI 
import os
import io 
import re 
import json
import glob
from pypdf import PdfReader
from .prompt import PARSE_AOR_PROMPT, PARSE_INVOICE_PROMPT
from .rule import Rule
from PIL import Image
import email
from email import policy


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


def get_pdf_contents(pdf_file, first_page=1, last_page=1):
    # Convert the first page of the PDF to an image
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)

    pdf_base64_images = []
    for pdf_image in images:
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        pdf_image.save(buffered, format="PNG")
        pdf_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        pdf_base64_images.append(pdf_image_base64)
        
    return pdf_base64_images


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


import ast 

def load_json_with_ast(json_str):
    json_str_cleaned = json_str.strip()
    papers = ast.literal_eval(json_str_cleaned)
    return papers


def parse_json_response(content):
    try:
        # Try to parse the entire content as JSON
        json_data = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON object using regex
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    json_data = load_json_with_ast(json_str)
                except:
                    return {}
        else:
            return {}
    
    return json_data

def read_eml(file_path):
    with open(file_path, 'rb') as file:
        msg = email.message_from_binary_file(file, policy=policy.default)
    
    # Extract basic information
    sender = msg['From']
    recipient = msg['To']
    subject = msg['Subject']
    date = msg['Date']

    # Get the email body
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                body = part.get_content()
                break
    else:
        body = msg.get_content()

    # Get attachments
    attachments = []
    for part in msg.iter_attachments():
        attachments.append(part.get_filename())

    return {
        'sender': sender,
        'recipient': recipient,
        'subject': subject,
        'date': date,
        'body': body,
        'attachments': attachments
    }


def preprocess_aor(aor_dir: str = "database/aor"):
    pdf_paths = glob.glob(f"{aor_dir}/*.pdf")
    eml_paths = glob.glob(f"{aor_dir}/*.eml")
    print("Preprocessing AORs...")
    for file_path in pdf_paths + eml_paths:
        # TBD: Skip preprocessing if things are already preprocessed
        if file_path.endswith(".pdf"):
            pdf_txt = get_pdf_text(file_path)
        elif file_path.endswith(".eml"):
            email_data = read_eml(file_path)
            pdf_txt = email_data.get('body')
        else:
            raise ValueError("Unknown file format")

        prompt = PARSE_AOR_PROMPT.format(pdf_txt=pdf_txt)
        max_tries = 3
        aor = None
        while not aor and max_tries > 0:
            response = get_oai_response(prompt)
            parsed_aor_dict = parse_json_response(response)
            try:
                aor = AOR(**parsed_aor_dict)
                aor.pdf_text = pdf_txt
                aor.pdf_path = file_path
                aor.save(aor_dir)
            except:
                max_tries -= 1
                continue
        
        if not aor:
            print(f"Failed to parse AOR: {file_path}")
            





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


def preprocess_invoice(invoice_dir: str = "database/invoice"):
    invoice_paths = [file for file in glob.glob(f"{invoice_dir}/*") if not file.endswith(".json")]

    print("Preprocessing Invoices...")
    for invoice_path in invoice_paths:
        if invoice_path.endswith(".pdf"):
            img = get_pdf_contents(invoice_path)[0]
        elif invoice_path.endswith(".png") or invoice_path.endswith(".jpg") or invoice_path.endswith(".jpeg"):
            with Image.open(invoice_path) as img_raw:
                buffered = io.BytesIO()
                img_raw.save(buffered, format="PNG")
                img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            raise ValueError("Unknown file format")
        
        max_tries = 3
        invoice = None
        while not invoice and max_tries > 0:
            response = get_oai_response(PARSE_INVOICE_PROMPT, img=img, img_type="base64")
            parsed_invoice_dict = parse_json_response(response)
            try:
                invoice = Invoice(**parsed_invoice_dict)
            except:
                max_tries -= 1
                continue
        
        if invoice:
            invoice.invoice_path = invoice_path
            invoice.save(invoice_dir)
        else:
            print(f"Failed to parse invoice: {invoice_path}")