# Retrieve rules from database
import glob
import pandas as pd
from typing import List, Tuple
import numpy as np
import re 
from sentence_transformers import SentenceTransformer
from .rule import Rule, load_rules, rules_to_string
from .utils import get_pdf_contents, get_oai_response


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
    



# def get_relevant_content(rule: Rule, pdf_path: str) -> str:
#     start_page, end_page = rule.page_range
#     content = get_pdf_contents(pdf_path, first_page=start_page, last_page=end_page)
#     return content

# def generate_augmented_answer(question: str, retrieved_rules: List[Tuple[Rule, float]], pdf_path: str) -> str:
#     context = ""
#     for rule, score in retrieved_rules:
#         context += f"Rule {rule.no} (Relevance: {score:.2f}):\n"
#         context += get_relevant_content(rule, pdf_path) + "\n\n"

#     prompt = f"""Given the following context from relevant rules, answer the question accurately and concisely.

# Context:
# {context}

# Question: {question}

# Answer:"""

#     return get_oai_response(prompt)

# def evaluate_answer(generated_answer: str, reference_answer: str) -> float:
#     prompt = f"""Compare the generated answer with the reference answer and rate the generated answer's accuracy on a scale of 0 to 1, where 1 is perfectly accurate and 0 is completely inaccurate or irrelevant.

# Generated Answer: {generated_answer}

# Reference Answer: {reference_answer}

# Accuracy Score (0-1):"""

#     response = get_oai_response(prompt)
#     try:
#         score = float(response.strip())
#         return max(0, min(1, score))  # Ensure the score is between 0 and 1
#     except ValueError:
#         print(f"Error parsing accuracy score: {response}")
#         return 0

# def process_qa_pair(question: str, reference_answer: str, rule_retriever: RuleRetriever, pdf_path: str) -> dict:
#     """
#     - Retrieve Rules
#     - Generated IGP answer 
#     - IGP-Evaluate on answer
#     """
#     retrieved_rules = rule_retriever.retrieve(question)
#     generated_answer = generate_augmented_answer(question, retrieved_rules, pdf_path)
#     accuracy_score = evaluate_answer(generated_answer, reference_answer)

#     return {
#         "question": question,
#         "reference_answer": reference_answer,
#         "generated_answer": generated_answer,
#         "accuracy_score": accuracy_score,
#         "retrieved_rules": [{"no": rule.no, "score": score} for rule, score in retrieved_rules]
#     }