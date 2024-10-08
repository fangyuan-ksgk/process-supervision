import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from dataclasses import field
import glob 
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import io

def pdf_to_img(pdf_file, first_page=1, last_page=1):
    # Convert the specified page range of the PDF to an image
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)        
    return images[0]

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

def load_rules(rule_dir: str = "database/rule"):
    rule_files = glob.glob(f"{rule_dir}/*.json")
    rule_list = []
    for rule_file in rule_files:
        rule = Rule.load(rule_file)
        rule_list.append(rule)
    return rule_list