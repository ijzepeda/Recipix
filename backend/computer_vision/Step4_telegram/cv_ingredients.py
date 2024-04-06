import random
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import ast
from segment_anything import  SamAutomaticMaskGenerator
import os
import torch

print(torch.cuda.is_available())

# Configuration and Model Paths
FOOD_CLASSIF_MODEL = "../models/model_FOOD_24-03_23-36_LR0.001.h5"
FOOD_CLASSIF_MODEL_CLASSES_FILE = "../models/model_FOOD_24-03_23-36_LR0.txt"
SAM_CHECKPOINT = "../models/sam_vit_h_4b8939.pth" #not on ../ but ./

SAM_CHECKPOINT_LOCAL = "C:\\Users\\ijzep\\_AI_\\CAPSTONE\\models\\sam_vit_h_4b8939.pth"

MODEL_TYPE = "vit_h"
DEVICE = "cuda"  # Use "cpu" for non-CUDA devices

# Check if the file exists at the SAM_CHECKPOINT path
if os.path.exists(SAM_CHECKPOINT):
    print(f"Using SAM checkpoint from: {SAM_CHECKPOINT}")
else:
    # If not, use the local version
    SAM_CHECKPOINT = SAM_CHECKPOINT_LOCAL
    print(f"File not found at {SAM_CHECKPOINT}. Using local version: {SAM_CHECKPOINT_LOCAL}")

def get_model_config(model_file=FOOD_CLASSIF_MODEL,
                     classes_file=FOOD_CLASSIF_MODEL_CLASSES_FILE,
                     sam_checkpoint_file=SAM_CHECKPOINT,
                     model_type=MODEL_TYPE,
                     device=DEVICE):

    return {
        "FOOD_CLASSIF_MODEL": f"{model_file}",
        "FOOD_CLASSIF_MODEL_CLASSES_FILE": f"{classes_file}",
        "SAM_CHECKPOINT": f"{sam_checkpoint_file}",
        "MODEL_TYPE": model_type,
        "DEVICE": device
    }



# Singleton Pattern for Model Loading
class SingletonModelLoader:
    _food_classif_model = None
    _sam_model = None
    _food_classes = None
    _sam_mask_generators = {}  # Use a dictionary to store instancesThe current implementation only checks if _sam_mask_generator is None before initializing it, without considering the value of detailed. This means that once the _sam_mask_generator is initialized, subsequent calls to get_sam_mask_generator with a different detailed value won't change the generator's configuration. For instance, if the generator is first initialized with detailed=False, then later calls with detailed=True will still return the generator configured for detailed=False.

    @classmethod
    def get_food_classif_model(cls):
        if cls._food_classif_model is None:
            cls._food_classif_model = load_model(FOOD_CLASSIF_MODEL)
        return cls._food_classif_model

    @classmethod
    def get_sam_model(cls):
        if cls._sam_model is None:
            from segment_anything import sam_model_registry
            cls._sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            cls._sam_model.to(device=DEVICE)
        return cls._sam_model

    @classmethod
    def get_food_classes(cls):
        if cls._food_classes is None:
            with open(FOOD_CLASSIF_MODEL_CLASSES_FILE) as f:
                data = f.read()
            cls._food_classes = ast.literal_eval(data)
        return cls._food_classes

    @classmethod
    def get_sam_mask_generator(cls, detailed=False, mask_detail="low"):
        #MAybe this one is saving the state
        # key = 'high' if detailed else 'low'
        key = mask_detail

        if key not in cls._sam_mask_generators:
            sam_model = cls.get_sam_model()
            if mask_detail=="low":
                # 30 segundos
                cls._sam_mask_generators[key] = SamAutomaticMaskGenerator(sam_model)
            elif mask_detail=="med":
                # 50 segundos=============================
                cls._sam_mask_generators[key] = SamAutomaticMaskGenerator(
                    model=sam_model,
                    points_per_side=32,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=250  # Requires open-cv to run post-processing
                )
            elif mask_detail=="high":
                # 80 segundos======================
                cls._sam_mask_generators[key] = SamAutomaticMaskGenerator(
                    model=sam_model,
                    points_per_side=32,
                    pred_iou_thresh=0.75,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=1,
                    min_mask_region_area=300  # Requires open-cv to run post-processing
                )
        return cls._sam_mask_generators[key]




def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def save_show_anns(img_color, anns, file_path='user_data/masked_image.png'):
    if len(anns) == 0:
        return img_color

    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    overlay = np.zeros_like(img_color)

    for ann in sorted_anns:
        segmentation_mask = ann['segmentation'].astype(bool)
        color_mask = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        overlay[segmentation_mask] = color_mask

    alpha = 0.35
    cv2.addWeighted(src1=img_color, alpha=1, src2=overlay, beta=alpha, gamma=0, dst=img_color)

    # Before saving, convert the image from BGR to RGB.
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_path, img_color_rgb)

    return file_path


def save_show_anns_plt(img_color, anns , file_path):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Assuming the shape of segmentation is compatible with being directly used as a mask
    img_height, img_width = sorted_anns[0]['segmentation'].shape[:2]
    img = np.ones((img_height, img_width, 4))  # Creating an RGBA image
    img[:, :, 3] = 0  # Making the background completely transparent

    for ann in sorted_anns:
        segmentation_mask = ann['segmentation']  # This assumes segmentation is a boolean numpy array
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # Adding some transparency
        img[segmentation_mask] = color_mask  # Apply the color mask where segmentation is True

    plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)  # Adjusting the figure size
    # plt.imshow(img)
    # plt.axis('off')  # Hide axes

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=100)  # Save the figure
    plt.close()  # Close the plot to free up memory

    return file_path  # Optional: return the path of the saved image


def preprocess_image(img, debug=False):
    # the code that was here to convert img color, and crop, where affecting the prediction
    if (debug):
        _img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        randvar = random.randint(50000, 100000)
        cv2.imwrite(f'debug/debug_image-{randvar}.jpg',_img)  # , cv2.COLOR_RGB2BGR

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def plot_save_masks(img_color, _masks , file_path='user_data/masks_annotations.png'):

    save_show_anns(img_color, _masks , file_path)  # colores de las mascaras

    return file_path

def _image_classifier(_image, _CONFIDENCE_THRESH=0.75,debug=False):
    preprocessed_image = preprocess_image(_image, debug)
    prediction = food_model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < _CONFIDENCE_THRESH:
        return "unknown"
    else:
        print("Predicted class:", predicted_class, index_to_class[predicted_class], confidence)
        return index_to_class[predicted_class]


food_model=None
index_to_class=None
def get_ingredients(image_path, detailed=False,_CONFIDENCE_THRESH=0.75, debug=False):
    print("this is the image to be classified", image_path)
    global food_model
    global index_to_class
    food_model = SingletonModelLoader.get_food_classif_model()
    index_to_class = SingletonModelLoader.get_food_classes()

    ingredients_found=[]
    def _process_found_bbox(_msk):
        # poner un limite de pequeñez y grandez de un objeto, # un objeto no puede abarcar toda la foto, un 50%? 75%
        x, y = img_color.shape[:2]
        MIN_HEIGHT = y * 0.05  # 0.1 #50
        MIN_WIDTH = x * 0.05  # 0.1 #50
        MAX_HEIGHT = y * 0.8  # 0.75 #545
        MAX_WIDTH = x * 0.8  # 0.75 #512


        print(f"Amount of masks/objects: {len(_msk)}")
        count = 0
        ingredients = []
        for _m in _msk:
            x1, y1, w, h = _m['bbox']  # XYWH
            if ((w >= MIN_WIDTH and h >= MIN_HEIGHT) and (w <= MAX_WIDTH and h <= MAX_HEIGHT)):
                count += 1
                cropped__img = img_color[y1:y1 + w, x1:x1 + h, :]
                # resize
                resized_img = cv2.resize(cropped__img, (224, 224))
                _ingredient = _image_classifier(resized_img,_CONFIDENCE_THRESH=_CONFIDENCE_THRESH , debug=debug)
                if _ingredient:
                    ingredients.append(_ingredient)

        print("Total masks of interest:", count)
        return ingredients

    img = cv2.imread(image_path)
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sam_mask_generator = SingletonModelLoader.get_sam_mask_generator(detailed, mask_detail='low')

    _masks = sam_mask_generator.generate(img_color)
    ingredients_found = _process_found_bbox(_masks)
    print(ingredients_found)

    # new_img_path = overlay_masks_on_image(img_color, _masks, "overlayed_image.jpg")
    # print("overlay_mask",new_img_path)
    #found masks, plot save send
    if debug:
        new_img_path = plot_save_masks(img_color, _masks)#, image_path
        print("plot_save mask",new_img_path)

    ingredients_found = set(ingredients_found)
    print(ingredients_found)
    ingredients_found.remove('unknown')
    return ingredients_found # new_img_path


#
# # Made with <3
# # by Ivan Zepeda
# # github@ijzepeda-LC