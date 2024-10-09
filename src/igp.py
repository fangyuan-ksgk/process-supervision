# Iterative Graph Prompting
import anthropic
import json 
import re, os
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import io
import base64
import subprocess

client = anthropic.Anthropic()

def prepare_graph_prompt(query: str, initial: bool = True):
    """
    Prepare the prompt for Iterative Graph Prompting
    """
    if initial:
        initial_prompt = f"""Given query: {query}

        Create a logical graph to map your thinking process. Apply the Occam's Razor principle to keep the graph as simple as possible while still capturing the essential reasoning. Provide your response in JSON format with 'nodes' and 'edges'.

        Example:
        {{
            "nodes": ["A", "B", "C"],
            "edges": [
                {{"from": "A", "to": "B", "relationship": "relates to"}},
                {{"from": "B", "to": "C", "relationship": "leads to"}}
            ]
        }}

        Please provide a similar JSON structure for the given query, focusing on the most essential elements and relationships."""
        return initial_prompt
        
    else:
        enhance_prompt = f"""Given query: {query}

        Analyze the diagram and update the graph to improve the reasoning process. Apply Occam's Razor principle to simplify the graph where possible. Consider removing unnecessary elements, refining existing ones, or addressing any gaps while maintaining simplicity.

        Provide your response in JSON format with 'nodes' and 'edges', like this:

        {{
            "nodes": ["A", "B", "C"],
            "edges": [
                {{"from": "A", "to": "B", "relationship": "leads to"}},
                {{"from": "B", "to": "C", "relationship": "influences"}}
            ]
        }}

        Please provide an updated graph based on your analysis, ensuring it remains as simple and concise as possible while still capturing the core reasoning."""
        return enhance_prompt


def get_claude_response(query, img = None, img_type = None, system_prompt = "You are a helpful assistant."):
    """ 
    Claude response with query and image input
    """
    # client = anthropic.Anthropic()
    if img is not None:
        text_content = [{"type": "text", "text": query}]
        img_content = [{"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img}}]
    else:
        text_content = query
        img_content = ""
        
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": img_content + text_content,
            }
        ],
        system=system_prompt,
    )
    return message.content[0].text 


def parse_json_from_response(response):
    """
    Parse out the JSON output from the 'response' string.
    
    Args:
    response (str): The string containing the JSON output.
    
    Returns:
    dict: The parsed JSON data, or None if parsing fails.
    """

    # Find the JSON content using regex
    json_match = re.search(r'\{[\s\S]*\}', response)
    
    if json_match:
        json_str = json_match.group(0)
        advice_str = response.split(json_str)[-1]
        try:
            # Parse the JSON string
            json_data = json.loads(json_str)
            
            # Ensure the JSON structure is as expected
            if 'nodes' in json_data and 'edges' in json_data:
                # No need to modify nodes or edges
                return json_data, advice_str
            else:
                print("Unexpected JSON structure.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON content found in the response.")
        return None
    
    

def build_graph_from_json(parsed_json):
    """ 
    Construct graph object from parsed json
    - nodes & edges as the key values
    """

    G = nx.DiGraph()

    # Add nodes
    for node in parsed_json['nodes']:
        try:
            G.add_node(node)
        except:
            G.add_node(node['id'], label=node['label'])
            
    # Add edges
    for edge in parsed_json['edges']:
        G.add_edge(edge['from'], edge['to'], label=edge['relationship'])

    return G 
    
    
def parse_logical_graph(response):
    """ 
    Parse out the Logical Graph Adopted by the LLM 
    """
    try:
        parsed_json, advice_str = parse_json_from_response(response)
        if parsed_json is None:
            print("Error: parse_json_from_response returned None.")
            return None, ""
        logical_graph = build_graph_from_json(parsed_json)
        return logical_graph, advice_str
    except json.JSONDecodeError:
        print("Error: Unable to parse the logical graph response as JSON.")
    except KeyError as e:
        print(f"Error: Missing key in the JSON structure: {e}")
    except nx.NetworkXError as e:
        print(f"Error: NetworkX error while building graph: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, ""


def get_logical_graph(query):
    """ 
    Under instruction, directly ask for CoT logical graph
    """
    igp_prompt = prepare_graph_prompt(query, initial=True)

    txt = get_claude_response(igp_prompt)

    # print("Test before parsing: ", txt)
        
    logical_graph, advice_str = parse_logical_graph(txt) # bug here

    return logical_graph


# D2 Diagram util function

d2_prefix = """
vars: {
  d2-config: {
    sketch: true
  }
}
classes: {
  process: {
    label: ""
    shape: rectangle
    style: {
      fill: lightblue
      shadow: true
    }
  }
}
"""

object_template = """{object_name}.class: process
{object_name}.label: "{object_label}"
{object_name}: {{
  style: {{
    opacity: 1
    stroke: "black"
    stroke-width: 2
    shadow: true
  }}
}}"""

link_template = """{start_object_id} -> {end_object_id}: {{
  label: "{relationship}"
  style.stroke: black
  style.opacity: 1
  style.stroke-width: 2
}}"""

def json_to_d2(json_data):
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    d2_output = [d2_prefix]
    
    # Add nodes
    for node in data['nodes']:
        object_name = node.replace(" ", "_").replace("'", "")
        d2_output.append(object_template.format(object_name=object_name, object_label=node))
    
    # Add edges
    for edge in data['edges']:
        from_node = edge['from'].replace(" ", "_").replace("'", "")
        to_node = edge['to'].replace(" ", "_").replace("'", "")
        relationship = edge['relationship']
        d2_output.append(link_template.format(start_object_id=from_node, end_object_id=to_node, relationship=relationship))
    
    return '\n'.join(d2_output)


def save_png_from_d2(d2_code, file_name, output_dir="d2_output"):
    """
    Save the d2_code as a .d2 file and generate the corresponding .svg file.
    
    Args:
    d2_code (str): The D2 diagram code.
    file_name (str): The base name for the output files (without extension).
    output_dir (str): The directory to save the files in.
    
    Returns:
    str: The path to the saved PNG file, or None if an error occurred.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the .d2 file
    d2_file_path = os.path.join(output_dir, f"{file_name}.d2")
    with open(d2_file_path, "w") as f:
        f.write(d2_code)
    
    # Generate the PNG file using the d2 command-line tool
    png_file_path = os.path.join(output_dir, f"{file_name}.png")
    try:
        subprocess.run(["d2", d2_file_path, png_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating PNG: {e}")
        png_file_path = None
    except FileNotFoundError:
        print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
        png_file_path = None
    
    return png_file_path

    
def remove_empty_edge(graph):
    """ 
    Remove empty edges (with no label) from the graph
    """
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        if 'label' not in data or not data['label']:
            G.remove_edge(u, v)
    return G


    
def store_graph(folder_dir, file_name, graph):
    """ 
    Store the graph into a file using D2 diagram format
    """
    # Convert graph to JSON format
    json_data = {
        "nodes": list(graph.nodes()),
        "edges": [{"from": u, "to": v, "relationship": data.get("label", "")} 
                  for u, v, data in graph.edges(data=True)]
    }
    
    # Convert JSON to D2 format
    d2_code = json_to_d2(json_data)
    
    # Save as PNG using D2
    png_file_path = save_png_from_d2(d2_code, file_name, output_dir=folder_dir)
    
    if png_file_path:
        print(f"Graph stored as PNG: {png_file_path}")
    else:
        print("Failed to store graph as PNG")
        
        
def visualize_graph(graph: nx.DiGraph):
    """ 
    Visualize D2 Diagram from the given graph
    """
    # convert graph to json
    json_data = {
        "nodes": list(graph.nodes()),
        "edges": [{"from": u, "to": v, "relationship": data.get("label", "")} 
                  for u, v, data in graph.edges(data=True)]
    }
    d2_code = json_to_d2(json_data)
    png_path = save_png_from_d2(d2_code, "test", output_dir="d2_output")
    
    if png_path:
        # Open the PNG file using Pillow
        img = Image.open(png_path)
        # Display the image
        img.show()
    else:
        print("Failed to generate or save the PNG file.")
        
def query_to_filename(query: str):
    """ 
    Convert query to a valid filename by removing or replacing invalid characters
    """
    # Remove any non-alphanumeric characters except hyphens and underscores
    sanitized = re.sub(r'[^\w\-]', '', query)
    # Replace spaces with hyphens
    sanitized = sanitized.replace(' ', '-')
    # Convert to lowercase
    sanitized = sanitized.lower()
    # Trim to a reasonable length (e.g., 50 characters)
    sanitized = sanitized[:50]
    # Ensure the filename is not empty
    if not sanitized:
        sanitized = "unnamed-query"
    return sanitized

def preprocess_image(query: str, graph: nx.DiGraph):
    """ 
    - Store query-specific heuristic graph under folder specific to the feedback
    - Load Image to base64 | Same for GPT-4o & Claude at least
    """
    
    # Store
    img_folder = f"database/evaluation/images"
    os.makedirs(img_folder, exist_ok=True)
    file_name = query_to_filename(query)
    store_graph(img_folder, file_name, graph) 
    
    # Load Image
    png_file_path = os.path.join(img_folder, f"{file_name}.png")
    with open(png_file_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    image_media_type = "image/png"
    return image_base64, image_media_type


def combine_images(diagram_img, input_img: str, num_pages: int = 1):
    """ 
    Combine diagram image and input image into one image, side by side
    """
    def base64_to_pil(base64_str):
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))

    diagram_pil = base64_to_pil(diagram_img)
    input_pil = base64_to_pil(input_img)

    # Calculate the target width for both images
    target_diagram_width = input_pil.width // num_pages
    target_height = input_pil.height
    
    # Diagram needs RESIZING to match target_diagram_width (but keep its aspect ratio)
    target_diagram_height = int(diagram_pil.height * target_diagram_width / diagram_pil.width)
    diagram_pil = diagram_pil.resize((target_diagram_width, target_diagram_height), Image.LANCZOS)
    
    total_width = target_diagram_width + input_pil.width

    # Create a new white image with the combined width and max height
    combined_img = Image.new('RGB', (total_width, target_height), color='white')

    # Paste the images side by side
    combined_img.paste(diagram_pil, (0, (target_height - target_diagram_height) // 2))
    combined_img.paste(input_pil, (target_diagram_width, 0))

    # Convert back to base64
    buffer = io.BytesIO()
    combined_img.save(buffer, format="PNG")
    combined_img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return combined_img_base64, combined_img


def enhance_logical_graph(query, graph):
    """ 
    Enhance logical Graph with IGP
    """
    igp_prompt = prepare_graph_prompt(query, initial=False)
    
    img, img_type = preprocess_image(query, graph)
    try:
        response = get_claude_response(igp_prompt, img, img_type)
        # print("Response: ", response)
        enhanced_logical_graph, advice_str = parse_logical_graph(response)
        if enhanced_logical_graph:
            return enhanced_logical_graph, advice_str
        else:
            return graph, ""  # Return original graph if enhancement fails
    except Exception as e:
        print(f"Error enhancing logical graph: {str(e)}")
        return graph, ""  # Return original graph when an error occurs
    
    

# ... existing code ...

def prepare_graph_query_prompt(query: str):
    """
    Prepare a prompt to query the graph for an answer using visual prompting
    """
    prompt = f"Answer the query: '{query}'. Use the graph in the image to guide your reasoning."
    
    return prompt

def graph_based_answer(query: str, graph: nx.DiGraph):
    """
    Generate an answer based on the graph structure using visual prompting
    """
    prompt = prepare_graph_query_prompt(query)
    img, img_type = preprocess_image(query, graph)
    response = get_claude_response(prompt, img, img_type)
    return response
    
def igp(query: str): 
    """
    Iterative Graph Prompting
    """
    initial_graph = get_logical_graph(query)
    enhanced_graph, _ = enhance_logical_graph(query, initial_graph)
    
    final_answer = graph_based_answer(query, enhanced_graph)
    
    return final_answer, enhanced_graph

# Variant of IGP required : it needs to take in extra image input from retrieved documents 

def prepare_image_grounded_graph_prompt(query: str):
    """
    Prepare the prompt for creating a logical graph grounded in the input image
    """
    prompt = f"""Given the query: '{query}'
    
    Analyze the provided image, which contains relevant information to answer the query.
    Create a logical graph that maps the thinking process to answer the query, using information from the image.
    Apply the Occam's Razor principle to keep the graph as simple as possible while capturing the essential reasoning.
    
    Provide your response in JSON format with 'nodes' and 'edges'.
    
    Example:
    {{
        "nodes": ["A", "B", "C"],
        "edges": [
            {{"from": "A", "to": "B", "relationship": "relates to"}},
            {{"from": "B", "to": "C", "relationship": "leads to"}}
        ]
    }}
    
    Ensure that the nodes and relationships in your graph are directly related to the content of the image and the query."""
    
    return prompt


def igp_with_image(query: str, input_image_base64: str, num_pages: int = 1, input_image_type: str= "image/png", system_prompt: str = "You are a helpful assistant."):

    # Generate initial graph grounded in the image
    initial_prompt = prepare_image_grounded_graph_prompt(query)
    initial_response = get_claude_response(initial_prompt, img=input_image_base64, img_type=input_image_type, system_prompt=system_prompt)

    # Parse the logical graph from the response
    logical_graph, _ = parse_logical_graph(initial_response)

    # Prepare the diagram
    diagram_img, _ = preprocess_image(query, logical_graph)

    # Combine Image
    combined_img_base64, combined_img = combine_images(diagram_img, input_image_base64, num_pages)

    # Visual Prompting 
    answer_prompt = f"""Answer the query: '{query}'.
    Use the information from both the reasoning graph and the original document shown in the image to provide a comprehensive answer."""

    # Get the final answer using both the diagram and input image
    response = get_claude_response(
        answer_prompt,
        img=combined_img_base64,
        img_type="image/png"
    )
    
    return response, logical_graph, combined_img