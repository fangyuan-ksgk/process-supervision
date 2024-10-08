# Rule Parsing Agent
from .utils import get_pdf_page_count, get_pdf_contents, get_oai_response, extract_json_from_content
from .prompt import PARSE_RULE_PROMPT, PARSE_AGENT_SYSTEM_PROMPT
import json
from typing import Optional
from .rule import Rule
def parse_rules_from_page_range(pdf_path, page_range):
    start_page, end_page = page_range
    img = get_pdf_contents(pdf_path, first_page=start_page, last_page=end_page)
    prompt = PARSE_RULE_PROMPT
    response = get_oai_response(prompt, img=img)
    rule_list = extract_json_from_content(response)
    return rule_list

def merge_rules(rule_list, new_rule_list):
    """ 
    In case of duplicate rule.no, merge the page_range for the rule with the same rule.no
    """
    for new_rule in new_rule_list:
        for rule in rule_list:
            if rule['no'] == new_rule['no']:
                rule['page_range'] = (min(rule['page_range'][0], new_rule['page_range'][0]), max(rule['page_range'][1], new_rule['page_range'][1]))
                break
        else:
            rule_list.append(new_rule)
    return rule_list

def fill_file_path(rule_list, file_path):
    """ 
    Fill in the file_path for each rule in the rule_list
    """
    for rule in rule_list:
        rule['file_path'] = file_path
    return rule_list


def present_current_rule_list(rule_list):
    """ 
    Construct Prompt for ParserAgent from its current rule list
    """
    if len(rule_list) == 0:
        return "No rules found yet. Consider parsing document from scratch, do you want to do it chunk-wise with better precision, or just all at once?"
    
    simplified_list = []
    for rule in rule_list:
        simplified_list.append(f"Rule {rule['no']}: pages {rule['page_range'][0]}-{rule['page_range'][1]}")
    
    res_str = "\n".join(simplified_list)
    
    res_str += "\n\nNote: The document contains continuous rules. If there are rules like 1.2 and 2.4, "
    res_str += "there must also be rules 2.1, 2.2, 2.3, and possibly 1.3, etc. in between."
    
    res_str += "\n\nConsider which page ranges might contain the missing rules. "
    res_str += "Are there any gaps in the current list where additional rules could be found?"
    
    res_str += "\n\nDo you want to parse the remaining rules chunk-wise for better precision, or parse all at once?"
    
    return res_str


def rule_list_filter(rule_list):
    # Keep rules with '.' in the number and remove unnatural numbering
    # Remove wrong no. like 1.a (pattern should be 1.2.a instead of 1.a) so one .a shows up while there is only one number as well
    filtered_rules = []
    for rule in rule_list:
        if '.' in rule['no']:
            # Check if the rule number follows the natural pattern (e.g., 1.2, 1.2.3)
            parts = rule['no'].split('.')
            
            if len(parts) == 2 and not parts[-1].isdigit():
                continue 
            else:
                filtered_rules.append(rule)

    return filtered_rules


def convert_and_store_rules(rule_list, output_dir="database/rule"):
    converted_rules = []
    for rule_dict in rule_list:
        rule = Rule(
            no=rule_dict['no'],
            description=rule_dict['description'],
            tags=rule_dict['tags'],
            file_path=rule_dict['file_path'],
            page_range=tuple(rule_dict['page_range'])
        )
        rule.save(output_dir)
        converted_rules.append(rule)
    return converted_rules


class ParseAgent:
    def __init__(self, pdf_path: str, chunk_size: int = 5, max_steps: int = 5):
        self.pdf_path = pdf_path
        self.page_count = get_pdf_page_count(pdf_path)
        self.chunk_size = chunk_size
        self.max_steps = max_steps
        self.rule_list = []
        self.records = []
        print(f"ParseAgent initialized with PDF: {pdf_path}")
        print(f"Total pages: {self.page_count}")

    def process_command(self, user_input: Optional[str] = None):
        print(f"Processing command. User input: {user_input}")
        
        user_input_str = "\n\nUser Input: " + user_input if user_input else ""
        rule_list_str = present_current_rule_list(self.rule_list)
        prompt = f"{PARSE_AGENT_SYSTEM_PROMPT}{user_input_str}\n\nCurrent Rule List: {rule_list_str}\nTotal Pages: {self.page_count}\n\nProvide your response in JSON format."
        self.records.append({"user": prompt})
        
        response = get_oai_response(prompt)
        print("Received response from OpenAI")
        
        try:
            action = extract_json_from_content(response)
            print(f"Extracted action: {action}")
        except json.JSONDecodeError:
            action = {"error": "Failed to parse JSON from response"}
            print("Error: Failed to parse JSON from response")
        except Exception as e:
            action = {"error": f"An unexpected error occurred: {str(e)}"}
            print(f"Unexpected error: {str(e)}")

        if action['tool'] == 'parse_rules_from_range':
            print(f"Parsing rules from range: {action['page_range']}")
            record = self.parse_rules_from_page_range(action['page_range'])
        elif action['tool'] == 'stop':
            record = {"message": "Parsing complete"}
            print("Parsing complete. Stopping.")
            return True
        else:
            record = {"error": "Invalid tool specified"}
            print(f"Error: Invalid tool specified - {action['tool']}")
            
        self.records.append(record)
        print(f"Added record: {record}")
        
        return False
    
    def run(self):
        """ 
        Parsing Agent Main Loop
        """
        print("Parsing Agent Main Loop")
        self.static_run() # Static Parsing 
        for i in range(self.max_steps): # Agent-based Parsing
            if self.process_command():
                break

    def parse_rules_from_page_range(self, page_range):
        """ 
        Parse rules from a given page range
        """
        parsed_rules = parse_rules_from_page_range(self.pdf_path, page_range)
        self.rule_list = merge_rules(self.rule_list, parsed_rules)
        self.rule_list = fill_file_path(self.rule_list, self.pdf_path)
        print(f"After Paring, current rule_list: {self.rule_list}")
        return {"message": f"Added {len(parsed_rules)} new rules", "new_rules": parsed_rules}
    
    def static_run(self):
        """ 
        Static Strategy for Parsing 
        - Parse chunk_size pages, before moving to the next one, loop through the document
        - Complimentary to the agent-based parsing, to collect missing rules
        """
        chunk_size = 10
        for i in range(0, self.page_count, chunk_size):
            print(f"Parsing rules from range: {i}-{i + chunk_size}")
            parsed_rules = self.parse_rules_from_page_range((i, i + chunk_size))
            
    def save_rules(self, output_dir="database/rule"):
        converted_rules = convert_and_store_rules(self.rule_list, output_dir)
        return converted_rules