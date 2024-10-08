# Iterative Graph Prompting
import anthropic
import json 
import re, os
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
import subprocess

client = anthropic.Anthropic()

def prepare_graph_prompt(query: str, initial: bool = True):
    """
    Prepare the prompt for Iterative Graph Prompting
    """
    if initial:
        initial_prompt = f"""Given query: {query}

        Create a logical graph to map your thinking process. Provide your response in JSON format with 'nodes' and 'edges'.

        Example:
        {{
            "nodes": ["A", "B", "C"],
            "edges": [
                {{"from": "A", "to": "B", "relationship": "relates to"}},
                {{"from": "B", "to": "C", "relationship": "leads to"}}
            ]
        }}

        Please provide a similar JSON structure for the given query."""
        return initial_prompt
        
    else:
        enhance_prompt = f"""Given query: {query}

        Analyze the diagram and update the graph to improve the reasoning process. Consider adding missing elements, refining existing ones, or addressing any gaps.

        Provide your response in JSON format with 'nodes' and 'edges', like this:

        {{
            "nodes": ["A", "B", "C"],
            "edges": [
                {{"from": "A", "to": "B", "relationship": "leads to"}},
                {{"from": "B", "to": "C", "relationship": "influences"}}
            ]
        }}

        Please provide an updated graph based on your analysis."""
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
        from PIL import Image
        img = Image.open(png_path)
        
        # Display the image
        img.show()
    else:
        print("Failed to generate or save the PNG file.")

def preprocess_image(query: str, graph: nx.DiGraph):
    """ 
    - Store query-specific heuristic graph under folder specific to the feedback
    - Load Image to base64 | Same for GPT-4o & Claude at least
    """
    
    # Store
    img_folder = f"database/evaluation/images"
    os.makedirs(img_folder, exist_ok=True)
    file_name = query.replace(" ", "-")
    store_graph(img_folder, file_name, graph) 
    
    # Load Image
    png_file_path = os.path.join(img_folder, f"{file_name}.png")
    with open(png_file_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    image_media_type = "image/png"
    return image_base64, image_media_type


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
    
    
def igp(query: str): 
    """
    Iterative Graph Prompting
    """
    initial_graph = get_logical_graph(query)
    enhanced_graph, advice_str = enhance_logical_graph(query, initial_graph)
    
    return enhanced_graph, advice_str