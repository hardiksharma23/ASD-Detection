import dlib 
import cv2
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from modeling import VisionTransformer, CONFIGS
from cww_for_vit import process_labels_confidence
from openpyxl import Workbook
import os
from PIL import Image

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = CONFIGS['ViT-B_16']

# Load models
model_eyes = VisionTransformer(config, 256, zero_head=True, num_classes=2)
model_path = './model_checkpoint/eyes_checkpoint.bin'
state_dict = torch.load(model_path, map_location=device)
model_eyes.load_state_dict(state_dict)
model_eyes.to(device)
model_eyes.eval()

model_nose = VisionTransformer(config, 256, zero_head=True, num_classes=2)
model_path = './model_checkpoint/nose_checkpoint.bin'
state_dict = torch.load(model_path, map_location=device)
model_nose.load_state_dict(state_dict)
model_nose.to(device)
model_nose.eval()

model_lips = VisionTransformer(config, 256, zero_head=True, num_classes=2)
model_path = './model_checkpoint/lips_checkpoint.bin'
state_dict = torch.load(model_path, map_location=device)
model_lips.load_state_dict(state_dict)
model_lips.to(device)
model_lips.eval()

# Load face detector and landmark predictor
predictor = dlib.shape_predictor("./model_checkpoint/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Image transformations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.05, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def process_image(image_path, output_folder):
    """
    Process a single image and return the annotated image and final decision.
    
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder to save the annotated image.
    
    Returns:
        tuple: (annotated_image, final_decision)
            - annotated_image: The processed image with bounding boxes and labels (numpy array).
            - final_decision: The final decision string (e.g., "autistic High").
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    image_save = image.copy()
    faces = detector(image)

    if len(faces) == 0:
        raise ValueError("No face detected in the image")

    for face in faces:
        landmarks = predictor(image_save, face)
        
        # Get the bounding box coordinates for the entire face
        x_min_face_full = face.left()
        y_min_face_full = face.top()
        x_max_face_full = face.right()
        y_max_face_full = face.bottom()

        # Draw the bounding box around the entire face (blue color)
        cv2.rectangle(image_save, (x_min_face_full, y_min_face_full), 
                     (x_max_face_full, y_max_face_full), (255, 0, 0), 2)

        # For eyes
        x_min_face = min(landmarks.part(17).x, landmarks.part(41).x + 6, landmarks.part(19).x, landmarks.part(24).x, landmarks.part(26).x, landmarks.part(47).x + 6)
        y_min_face = min(landmarks.part(17).y, landmarks.part(41).y + 6, landmarks.part(19).y, landmarks.part(24).y, landmarks.part(26).y, landmarks.part(47).y + 6)
        x_max_face = max(landmarks.part(17).x, landmarks.part(41).x + 6, landmarks.part(19).x, landmarks.part(24).x, landmarks.part(26).x, landmarks.part(47).x + 6)
        y_max_face = max(landmarks.part(17).y, landmarks.part(41).y + 6, landmarks.part(19).y, landmarks.part(24).y, landmarks.part(26).y, landmarks.part(47).y + 6)
        
        # For nose
        x_min_nose = min(landmarks.part(39).x, landmarks.part(42).x, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(33).x)
        y_min_nose = min(landmarks.part(39).y, landmarks.part(42).y, landmarks.part(31).y, landmarks.part(35).y, landmarks.part(33).y)
        x_max_nose = max(landmarks.part(39).x, landmarks.part(42).x, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(33).x)
        y_max_nose = max(landmarks.part(39).y, landmarks.part(42).y, landmarks.part(31).y, landmarks.part(35).y, landmarks.part(33).y)
        
        # For lips
        x_min_lips = min(landmarks.part(48).x, landmarks.part(54).x, landmarks.part(50).x, landmarks.part(52).x, landmarks.part(57).x)
        y_min_lips = min(landmarks.part(48).y, landmarks.part(54).y, landmarks.part(50).y, landmarks.part(52).y, landmarks.part(57).y)
        x_max_lips = max(landmarks.part(48).x, landmarks.part(54).x, landmarks.part(50).x, landmarks.part(52).x, landmarks.part(57).x)
        y_max_lips = max(landmarks.part(48).y, landmarks.part(54).y, landmarks.part(50).y, landmarks.part(52).y, landmarks.part(57).y)
        
        crop_img_eyes = image[y_min_face:y_max_face, x_min_face:x_max_face]
        crop_img_nose = image[y_min_nose:y_max_nose, x_min_nose:x_max_nose]
        crop_img_lips = image[y_min_lips:y_max_lips, x_min_lips:x_max_lips]

        if crop_img_eyes.size == 0 or crop_img_nose.size == 0 or crop_img_lips.size == 0:
            raise ValueError("Failed to crop facial features")

        crop_img_eyes_copy = crop_img_eyes.copy()
        crop_img_nose_copy = crop_img_nose.copy()
        crop_img_lips_copy = crop_img_lips.copy()
        
        cv2.rectangle(image_save, (x_min_face, y_min_face), (x_max_face, y_max_face), (0, 255, 0), 2)
        cv2.rectangle(image_save, (x_min_nose, y_min_nose), (x_max_nose, y_max_nose), (0, 255, 255), 2)
        cv2.rectangle(image_save, (x_min_lips, y_min_lips), (x_max_lips, y_max_lips), (255, 255, 0), 2)

        # For eyes
        image = cv2.cvtColor(crop_img_eyes_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform_train(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_eyes = model_eyes(image)[0]
            logits_eyes = F.softmax(logits_eyes, dim=-1)

        # For nose
        image = cv2.cvtColor(crop_img_nose_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform_train(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_nose = model_nose(image)[0]
            logits_nose = F.softmax(logits_nose, dim=-1)

        # For lips
        image = cv2.cvtColor(crop_img_lips_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform_train(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_lips = model_lips(image)[0]
            logits_lips = F.softmax(logits_lips, dim=-1)

        # For eyes
        if logits_eyes[0][0] > 0.5:
            text = 'autistic'
            logit_face_print = logits_eyes[0][0]
            eyes = 'autistic'
        else:
            text = 'non-autistic'
            logit_face_print = (1 - logits_eyes[0][0])
            eyes = 'non-autistic'
        text += f' {logit_face_print*100:.2f}%'
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_width = text_width + 10
        rect_height = text_height + 20
        rect_x = max(0, x_min_face)
        rect_y = max(0, y_min_face - rect_height)
        rect_x_end = min(image_save.shape[1], rect_x + rect_width)
        rect_y_end = min(image_save.shape[0], rect_y + rect_height)
        cv2.rectangle(image_save, (rect_x, rect_y), (rect_x_end, rect_y_end), (255, 0, 255), -1)
        text_x = rect_x + int((rect_width - text_width) / 2)
        text_y = rect_y + int((rect_height + text_height) / 2)
        cv2.putText(image_save, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # For nose
        if logits_nose[0][0] > 0.5:
            text = 'autistic'
            logit_nose_print = logits_nose[0][0]
            nose = 'autistic'
        else:
            text = 'non-autistic'
            logit_nose_print = (1 - logits_nose[0][0])
            nose = 'non-autistic'
        text += f' {logit_nose_print*100:.2f}%'
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_width = text_width + 10
        rect_height = text_height + 20
        rect_x = max(0, x_min_nose)
        rect_y = max(0, y_min_nose - rect_height)
        rect_x_end = min(image_save.shape[1], rect_x + rect_width)
        rect_y_end = min(image_save.shape[0], rect_y + rect_height)
        cv2.rectangle(image_save, (rect_x, rect_y), (rect_x_end, rect_y_end), (0, 255, 255), -1)
        text_x = rect_x + int((rect_width - text_width) / 2)
        text_y = rect_y + int((rect_height + text_height) / 2)
        cv2.putText(image_save, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # For lips
        if logits_lips[0][0] > 0.5:
            text = 'autistic'
            logit_lips_print = logits_lips[0][0]
            lips = 'autistic'
        else:
            text = 'non-autistic'
            logit_lips_print = (1 - logits_lips[0][0])
            lips = 'non-autistic'
        text += f' {logit_lips_print*100:.2f}%'
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_width = text_width + 10
        rect_height = text_height + 20
        rect_x = max(0, x_min_lips)
        rect_y = max(0, y_min_lips - rect_height)
        rect_x_end = min(image_save.shape[1], rect_x + rect_width)
        rect_y_end = min(image_save.shape[0], rect_y + rect_height)
        cv2.rectangle(image_save, (rect_x, rect_y), (rect_x_end, rect_y_end), (255, 255, 0), -1)
        text_x = rect_x + int((rect_width - text_width) / 2)
        text_y = rect_y + int((rect_height + text_height) / 2)
        cv2.putText(image_save, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        logit_face_print_value = logit_face_print.item()
        logit_nose_print_value = logit_nose_print.item()
        logit_lips_print_value = logit_lips_print.item()

        labels = [eyes, nose, lips]
        confidences = [logit_face_print_value, logit_nose_print_value, logit_lips_print_value]

        # Call function from cww_for_vit.py to get the final decision
        final_decision = process_labels_confidence(labels, confidences)

        # Add the final decision label on the top-left corner of the face bounding box
        text = final_decision
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_width = text_width + 10
        rect_height = text_height + 20
        rect_x = max(0, x_min_face_full - 20)
        rect_y = max(0, y_min_face_full - rect_height - 40)
        rect_x_end = rect_x + rect_width
        rect_y_end = rect_y + rect_height
        if rect_y < 0:
            rect_y = 0
            rect_y_end = rect_height
        cv2.rectangle(image_save, (rect_x, rect_y), (rect_x_end, rect_y_end), (255, 255, 255), -1)
        text_x = rect_x + int((rect_width - text_width) / 2)
        text_y = rect_y + int((rect_height + text_height) / 2)
        cv2.putText(image_save, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    # Save the annotated image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image_save)

    return image_save, final_decision

if __name__ == "__main__":
    # For autistic image folder
    input_folder = './Data/Faces/test/autistic'
    output_folder = 'Results/autistic_output'
    os.makedirs(output_folder, exist_ok=True)

    filename_aut = 'output_autistic.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.append(['Actual', 'Predicted'])
    ws.append(['Filename', 'Eyes', 'Eyes_logits', 'Nose', 'Nose_logits', 'Lips', 'Lips_logits'])

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            print(filename)
            try:
                annotated_image, final_decision = process_image(image_path, output_folder)
                print("Final Decision from CWW:", final_decision)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # For non-autistic image folder
    input_folder = './Data/Faces/test/non_autistic'
    output_folder = 'Results/non_autistic_output'
    os.makedirs(output_folder, exist_ok=True)

    filename_nonaut = 'output_non_autistic.xlsx'
    wbb = Workbook()
    wss = wbb.active
    wss.append(['Actual', 'Predicted'])
    wss.append(['Filename', 'Eyes', 'Eyes_logits', 'Nose', 'Nose_logits', 'Lips', 'Lips_logits'])

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            print(filename)
            try:
                annotated_image, final_decision = process_image(image_path, output_folder)
                print("Final Decision from CWW:", final_decision)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
