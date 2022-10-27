from data_cleaning.clean_images import resize_image
from PIL import Image
from image_classifier_model import training_transforms
import torch

def process_image(image):
    '''
    parameters:
    ----------
    image: str
        The path to the image to be processed.
    '''
    im = Image.open(image)
    resized = resize_image(512, im)
    transformed_image = training_transforms(resized)
    unsqueezed_image = torch.unsqueeze(transformed_image, 0)
    return unsqueezed_image