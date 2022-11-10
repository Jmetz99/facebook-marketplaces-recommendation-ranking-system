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

## Building the Image Classifier
Importing the RESNET50 model from torch hub, an additional linear layer was added to alter the outputs image classification into 13 categories.

```
    self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, 13).to(device)
        self.main = nn.Sequential(self.resnet50, self.linear).to(device)
```

This model was then fit onto the training data with training performance exceeding 60% accuracy. 

## Building the Description Classifier
Firstly, the output of the BERT model was used to create embeddings for the product text descriptions:
```
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
     
     
    encoded = self.tokenizer.__call__([sentence], max_length=self.max_length, padding='max_length', truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
        description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
```

These embeddings were then fed into a convolutional neural network with 4 convolutional layers. Training performace exceeded 90% .
