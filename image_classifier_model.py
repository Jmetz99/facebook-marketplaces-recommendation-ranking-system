import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torchvision
import torch.utils.data 
from torchvision import datasets, models, transforms
import datetime
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
training_transforms = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

class FBImageClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, 13).to(device)
        self.main = nn.Sequential(self.resnet50, self.linear).to(device)

    def forward(self, X):
        return self.main(X)

def train(model, data_loader, epochs=10):
    writer = SummaryWriter()

    optimiser = torch.optim.SGD(model.parameters(), lr=0.002)

    criterion = nn.CrossEntropyLoss()

    batch_index = 0

    for epoch in range(epochs):
        accuracy = []
        pbar = tqdm(data_loader, total=len(data_loader))
        for batch in pbar:
            features, labels = batch
            features.to(device)
            labels.to(device)
            optimiser.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimiser.step()
            accuracy_batch = torch.sum(torch.argmax(predictions, dim=1) == labels).item()/len(labels)
            accuracy.append(accuracy_batch)
            ave_accuracy = np.mean(accuracy)
            writer.add_scalar('Loss', loss.item(), batch_index)
            pbar.set_description(str(ave_accuracy))
            batch_index += 1
        
        now = str(datetime.datetime.now().time())
        path = f'model_evaluation/image_model_evaluations/image_model_at_{now}_epoch:{epoch}'
        os.mkdir(path)
        torch.save(model.state_dict(), f'{path}/weights.pt')
        with open(f'{path}/accuracy_{now}.txt', 'w') as f:
            f.write(f'{ave_accuracy}')


if __name__ == '__main__':
    training_data_path = '/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/training_image_data'
    train_dataset = torchvision.datasets.ImageFolder(root=training_data_path, transform=training_transforms)
    training_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = FBImageClassifier()
    train(model, training_data_loader)