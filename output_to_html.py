from bs4 import BeautifulSoup
import json

def modify_html_field(file_path, field_id, new_content):
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Find the element with the specified ID
    element = soup.find(id=field_id)
    
    if element:
        # Modify the inner HTML of the element
        element.clear()
        new_content = BeautifulSoup(new_content, 'html.parser')
        element.append(new_content)
    else:
        print(f"No element found with ID '{field_id}'")
        return

    # Write the modified HTML back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))

# Example usage
file_path = 'output/Chat with Open Large Language Models.html'
data_path_a = "debug.json"
data_path_b = "debug2.json"

data_a = json.load(open(data_path_a, 'r'))
data_b = json.load(open(data_path_b, 'r'))

for i, (dp_a, dp_b) in enumerate(zip(data_a, data_b)):
    if i >= 3:
        break
    modify_html_field(file_path, "modela-text", dp_a)
    modify_html_field(file_path, "modelb-text", dp_b)
    
