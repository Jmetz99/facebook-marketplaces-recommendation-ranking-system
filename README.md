# Facebook Marketplace Recommendation Ranking System

## Cleaning product data and images
### Product data
Given that all project data except IDs were object types, the prices were stripped of £ signs and converted to float64 types:
```
    df["price"] = df["price"].str.strip('£')
    df["price"] = df["price"].str.replace(',', '')
    df["price"] = df["price"].astype('float64')
```
### Product Images
The product images were resized by first finding the ratio of the old image size to the desired image size (512):
```
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
```

Then, using Pillow again to resize the image with this ratio and the ANTIALIAS filter to reduce visual defects of lower resolution:`im = im.resize(new_image_size, Image.ANTIALIAS)`

Then, generating a new image and pasting the resized image within this, defining the box (upper left corner) as shown below, to arrive at the final resized image 'new_im':
```
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
``` 