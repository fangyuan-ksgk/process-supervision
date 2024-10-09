import pandas as pd
from tqdm import tqdm

def evaluate_rule_retrieval(qa_df, rule_retriever):
    correct_retrievals = 0
    total_questions = len(qa_df)

    for _, row in tqdm(qa_df.iterrows(), total=total_questions, desc="Evaluating rule retrieval"):
        question = row['Question']
        correct_rule_no = row['Rulebook Content No.']
        
        retrieved_rule = rule_retriever(question)
        retrieved_rule_no = retrieved_rule.no if retrieved_rule else None
        
        print(f"Predicted: {retrieved_rule_no}, Correct: {correct_rule_no}")
        
        if retrieved_rule_no == correct_rule_no:
            correct_retrievals += 1

    accuracy = correct_retrievals / total_questions
    print(f"Rule retrieval accuracy: {accuracy:.2%}")
    return accuracy