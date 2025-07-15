import os
from test_main import process_image
from openpyxl import Workbook

# Set up test folders (update these paths if your test images are elsewhere)
autistic_folder = './Data/Faces/test/autistic'
non_autistic_folder = './Data/Faces/test/non_autistic'
output_xlsx = 'classification_results.xlsx'

wb = Workbook()
ws = wb.active
ws.append(['Filename', 'Actual', 'Predicted'])

# Process autistic images
for filename in os.listdir(autistic_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(autistic_folder, filename)
        try:
            _, prediction = process_image(image_path, './Results/autistic_output')
            ws.append([filename, 'autistic', prediction])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Process non-autistic images
for filename in os.listdir(non_autistic_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(non_autistic_folder, filename)
        try:
            _, prediction = process_image(image_path, './Results/non_autistic_output')
            ws.append([filename, 'non-autistic', prediction])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

wb.save(output_xlsx)
print(f"Results saved to {output_xlsx}")