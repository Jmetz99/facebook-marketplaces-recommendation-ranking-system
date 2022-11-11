import pandas as pd
from PIL import Image
from numpy import asarray
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = '/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/images'
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs, 1):
        im = Image.open(f'{path}/' + item)
        new_im = resize_image(final_size, im)
        file_name = item.replace('.jpg', '_resized.jpg')
        new_im.save(f'/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/resized_images/{file_name}')
        