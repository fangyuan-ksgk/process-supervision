# Retrieve rules from database
import glob
import pandas as pd
from typing import List
import re 
from .rule import load_rules, rules_to_string
from .utils import get_oai_response


def load_qa_data(rulebook_no):
    qa_dir = "raw/qa"
    qa_file = glob.glob(f"{qa_dir}/*.xlsx")[0]
    qa_df = pd.read_excel(qa_file)
    qa_df = qa_df[qa_df['Rulebook No.'] == 1]  # extract info only for rule-book 1
    return qa_df


def extract_list_from_string(response_str):
    match = re.search(r'\[(.*?)\]', response_str)
    if match:
        extracted_list = match.group(1).split(', ')
    else:
        extracted_list = []
        print("No list found in the string.")
    return extracted_list

def retrieve_relevant_rules(query, rules_str):
    prompt = f"""Given the following query and a list of rules, identify the most relevant rule numbers. 
    Return your response as a JSON array of rule numbers.

    Query: {query}

    Rules:
    {rules_str}

    Most relevant 4 rule numbers:
    ```json
    """

    response = get_oai_response(prompt)
    return extract_list_from_string(response)


class RuleRetriever:
    def __init__(self, rule_dir: str = "database/rule/01._RB_1_General_Safety_Requirements_for_Access_to_Track_and_Protection_Methods.pdf"):
        self.rules = load_rules(rule_dir)

    def _retrieve_relevant_rules(self, query: str):
        rules_str = rules_to_string(self.rules)
        rules_no = retrieve_relevant_rules(query, rules_str)
        rules_from_no = [rule for rule in self.rules if rule.no in rules_no]
        return rules_from_no
    
    def __call__(self, query: str):
        relevant_rules = self._retrieve_relevant_rules(query)
        return relevant_rules[0] if len(relevant_rules) > 0 else None
