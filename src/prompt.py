SYSTEM_PROMPT = """You are Admin Agent developed by Temus. You are tasked with answering questions about AORs and invoices. 

You are given a user query and can retrieve AOR and invoice information to answer the user query. 
When you do not retrieve relevant AORs or Invoices, you could respond with "Sorry I cannot find the relevant information".
You are also happy to provide direct answer to chat with the user.
"""


PARSE_AOR_PROMPT = """Please analyze the AOR (Authority of Requirement) file and provide the following information in JSON format:

{{
    "items": ["list of items covered by the AOR"],
    "budgets": [0.0, 0.0, ...],  // list of corresponding budgets for each item
    "no": "AOR number",
    "description": "brief description of the AOR",
    "expiry_date": "YYYY-MM-DD"
}}

Here is the AOR file: {pdf_txt}"""



PARSE_INVOICE_PROMPT = """Please analyze the invoice image and provide the following information in JSON format:

{{
    "no": "invoice number",
    "date": "YYYY-MM-DD",
    "currency": "currency used (Yuan, Thai Baht, Singapore Dollar, or US Dollar)",
    "seller": "name of the seller", // translate to English if it's not already in English
    "items": ["list of items or services on the invoice"], // translate to English if it's not already in English
    "amounts": [0.0, 0.0, ...],  // list of corresponding amounts for each item
    "description": "brief description of the invoice", // translate to English if it's not already in English
    "invoice_text": "full text content of the invoice" // translate to English if it's not already in English
}}

Please extract this information from the provided invoice image. Use English only."""