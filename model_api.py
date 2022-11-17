import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_processor import process_image
from text_processor import process_text


class TextClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = 13,
                 input_size: int = 768
                 ):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_classes),
                                    )
    def forward(self, X):
        return self.layers(X)
        
    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            pass

    def predict_classes(self, text):
        with torch.no_grad():
            pass

class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, 13)
        self.main = nn.Sequential(self.resnet50, self.linear)

    def forward(self, X):
        return self.main(X)

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            pass

    def predict_classes(self, image):
        with torch.no_grad():
            pass


class CombinedModel(nn.Module):
    def __init__(self,
                 num_classes: int = 13,
                 input_size: int=768):
        super(CombinedModel, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.image_linear = nn.Linear(out_features, 128)
        self.image_model = nn.Sequential(self.resnet50, nn.ReLU(), self.image_linear, nn.ReLU())

        self.text_model = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64, 128),
                                    nn.ReLU())

        self.main = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, X: tuple):
        image_inputs = self.image_model(X[0])
        text_inputs = self.text_model(X[1])
        comb_inputs = torch.cat((image_inputs, text_inputs), 1)
        return self.main(comb_inputs)

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            pass

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            pass



# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
    text_model = TextClassifier()
    text_model.load_state_dict(torch.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/model_evaluation/text_model_evaluations/text_model_at_15:09:36.157586_epoch:3/weights.pt'))
    text_model.eval()
    decoder = pickle.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/multi_decoder.pkl')
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
    image_model = ImageClassifier()    
    image_model.load_state_dict(torch.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/model_evaluation/image_model_evaluations/image_model_at_16:17:16.561435.pt'))                         #
    image_model.eval()
    decoder = pickle.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/multi_decoder.pkl')
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    combined_model = CombinedModel()    
    combined_model.load_state_dict(torch.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/model_evaluation/multimodal_model_evaluations/model_&_opt/multimodal_model_at_10:11:54.165316_epoch:4/model_state_dict.pt'))                         #
    combined_model.eval()
    decoder = pickle.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/multi_decoder.pkl')
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(text: TextItem):
    processed_text = process_text(text)
    prediction = text_model.predict(processed_text)
    category = decoder[int(torch.argmax(prediction, dim=1))]

    return JSONResponse(content={
        "Category": f"{category}", 
        "Probabilities": '' # Return a list or dict of probabilities here
            })
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    processed_image = process_image(pil_image)
    prediction = image_model.predict(processed_image)
    category = decoder[int(torch.argmax(prediction, dim=1))]

    return JSONResponse(content={
    "Category": f"{category}", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    processed_image = process_image(pil_image)
    processed_text = process_text(text)
    prediction = combined_model.predict((processed_image, processed_text))
    category = decoder[int(torch.argmax(prediction, dim=1))]

    return JSONResponse(content={
    "Category": f"{category}", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)