# Rule Parsing Agent
from utils import get_pdf_page_count, get_pdf_contents, get_oai_response, extract_json_from_content
from prompt import PARSE_RULE_PROMPT, PARSE_AGENT_SYSTEM_PROMPT
import json

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


class ParseAgent:
    """
    ParseAgent is designed to interact with a rulebook PDF file, allowing for the addition, modification, and querying of rules within the document.
    - Ideal for tracking historical message for fine-tuning task-specific LLM Agent
    """
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.page_count = get_pdf_page_count(pdf_path)
        self.rule_list = []

    def process_command(self, user_input):
        prompt = f"{PARSE_AGENT_SYSTEM_PROMPT}\n\nUser Input: {user_input}\n\nCurrent Rule List: {json.dumps(self.rule_list, indent=2)}\n\nProvide your response in JSON format."
        response = get_oai_response(prompt)
        action = json.loads(response)

        if action['tool'] == 'parse_rules_from_range':
            result = self.parse_rules_from_page_range(action['page_range'])
        else:
            result = {"error": "Invalid tool specified"}

        return result

    def parse_rules_from_page_range(self, page_range):
        parsed_rules = parse_rules_from_page_range(self.pdf_path, page_range)
        self.rule_list = merge_rules(self.rule_list, parsed_rules)
        return {"message": f"Added {len(parsed_rules)} new rules", "new_rules": parsed_rules}