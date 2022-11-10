from text_loader import TextDatasetBert
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import datetime
import os
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TextClassifier(nn.Module):
    def __init__(self,
                 num_classes,
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

def train(model, dataloader, epochs=5,):
    writer = SummaryWriter()

    optimiser = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()



    batch_index = 0
    for epoch in range(epochs):
        accuracy = []
        pbar = tqdm(dataloader, total=len(dataloader))
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
            writer.add_scalar('Average Accuracy', ave_accuracy, batch_index)
            pbar.set_description(str(ave_accuracy))
            batch_index += 1
        
        now = str(datetime.datetime.now().time())
        path = f'model_evaluation/text_model_evaluations/text_model_at_{now}_epoch:{epoch}'
        os.mkdir(path)
        torch.save(model.state_dict(), f'{path}/weights.pt')
        with open(f'{path}/accuracy_{now}.txt', 'w') as f:
            f.write(f'{ave_accuracy}')


if __name__ == '__main__':
    dataset = TextDatasetBert()
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    num_classes = dataset.num_classes
    model = TextClassifier(num_classes)
    train(model, dataloader)