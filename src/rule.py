import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from dataclasses import field
import glob 
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import io

def pdf_to_img(file_path, first_page=1, last_page=1):
    # Convert PDF pages to images
    img_pages = convert_from_path(file_path, first_page=first_page, last_page=last_page)

    # Calculate the total width and maximum height
    total_width = sum(page.width for page in img_pages)
    max_height = max(page.height for page in img_pages)

    # Create a new image with the calculated dimensions
    combined_img = Image.new('RGB', (total_width, max_height))

    # Paste each page into the combined image
    x_offset = 0
    for page in img_pages:
        combined_img.paste(page, (x_offset, 0))
        x_offset += page.width

    return combined_img

def file_to_img(file_path, first_page=1, last_page=1):
    if file_path.endswith(".pdf"):
        img = pdf_to_img(file_path, first_page, last_page)
    elif file_path.endswith(".eml"):
        img = None 
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        with Image.open(file_path) as img_raw:
            buffered = io.BytesIO()
            img_raw.save(buffered, format="PNG")
            img = buffered.getvalue()
    else:
        raise ValueError("Unknown file format")
    return img


def file_to_preprocessed_img(file_path, first_page=1, last_page=1):
    import base64

    if file_path.endswith(".pdf"):
        img = pdf_to_img(file_path, first_page, last_page)
    elif file_path.endswith(".eml"):
        return None
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
    else:
        raise ValueError("Unknown file format")

    # Convert image to PNG
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    # Convert PNG to base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64


@dataclass
class Rule:
    no: str
    description: str
    tags: List[str]
    file_path: str
    page_range: tuple = (1, 1)  # Default to the first page

    @property
    def image(self):
        return file_to_img(self.file_path, *self.page_range)
    
    @property
    def preprocessed_image(self):
        """ 
        Used for Image Prompting
        Returns a tuple of (base64_encoded_image, media_type)
        """
        img_base64 = file_to_preprocessed_img(self.file_path, *self.page_range)
        return img_base64
        
    
    def save(self, rule_dir: str = "database/rule"):
        import os
        import json
        
        # Create the directory if it doesn't exist
        os.makedirs(rule_dir, exist_ok=True)
        
        self.no = self.no.replace("/","-")

        # Create the file path
        file_path = os.path.join(rule_dir, f"{self.no}.json")
        
        # Convert the Rule object to a dictionary
        rule_dict = dataclasses.asdict(self)
        
        # Write the dictionary to a JSON file
        with open(file_path, 'w') as f:
            json.dump(rule_dict, f, indent=4)

    @classmethod
    def load(cls, rule_path: str):
        import json
        with open(rule_path, "r") as f:
            rule_dict = json.load(f)
        rule_dict['no'] = rule_dict['no'].replace("-", "/")
        return Rule(**rule_dict)
    
    @property
    def narrative(self):
        tags_str = ", ".join(self.tags)
        narrative_str = f"Rule no. {self.no}\nDescription: {self.description}\nTags: {tags_str}\nFile Path: {self.file_path}\nPage Range: {self.page_range}"
        return narrative_str

def load_rules(rule_dir: str = "database/rule/01._RB_1_General_Safety_Requirements_for_Access_to_Track_and_Protection_Methods.pdf"):
    rule_files = glob.glob(f"{rule_dir}/*.json")
    rule_list = []
    for rule_file in rule_files:
        rule = Rule.load(rule_file)
        rule_list.append(rule)
    return rule_list

def rule_to_string(rule: Rule):
    return f"{rule.no} {rule.description} {rule.tags}"

def rules_to_string(rules: List[Rule]):
    # Sort the rules based on the 'no' attribute
    sorted_rules = sorted(rules, key=lambda r: r.no)
    return "\n".join([rule_to_string(rule) for rule in sorted_rules])