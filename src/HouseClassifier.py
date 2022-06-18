import numpy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from doodleLoaderSimple import DoodleDatasetSimple

'''
Training and validation for the house image classifier
'''

# Number to class labels mapping
class_dict = {
    0: 'stress',
    1: 'introvert',
    2: 'extrovert'
}

# Loading the data from the .csv file
# First row is a header
data = np.genfromtxt('../data/houseData.csv', dtype=int, delimiter=',', names=True)


def count_classes(dictClass, arr):
    """
    Redundant method that counts the occurrences of each class in the dataset
    Can be used to create weights if the class distribution is unbalanced
    :param dictClass: Dictionary that maps number to class labels
    :param arr: The array that contains the data
    :return: The number of occurrences for each class in the given array
    """
    unique, count = numpy.unique(arr, return_counts=True)
    print(dict(zip(dictClass.values(), count)))
    count = 1 / count
    count = count / sum(count)
    return count


# Match the image IDs to the ID values in the .csv file.
translation_dict = dict(
    zip([f'{id}.png' for id in data['id']], data['class']))

# Prepare each image to be passed as a Tensor product to the model.
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prepare the data by matching it to it's label and transforming it to a Tensor product.
housedata = DoodleDatasetSimple('../images/house/', data_transforms, translation_dict)

# 80% of the data for training.
train_len = int(housedata.__len__() * 0.8)
# 20% of the data for validation.
test_len = int(housedata.__len__() * 0.2 + 1)
# Split the data at a random point.
train_set, val_set = torch.utils.data.random_split(housedata, [train_len, test_len])
# Shuffle and load the labeled images in batches of 4 for training.
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
# Load the labeled images in batches of 4 for validation after training the model.
test_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0, drop_last=True)


class MultilabelClassifier(nn.Module):
    """
    Class that contains the layers for the model.
    Starting model ResNet-34, replace last layer with a Linear layer that outputs
    a single number, the label of the image.
    """
    def __init__(self, n_features):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.imageClass = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=n_features)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'class': self.imageClass(x)
        }


# Set the device to use as the GPU if there is compatible hardware
# Otherwise run the model on the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultilabelClassifier(3).to(device)


def criterion(outputs, pictures):
    """
    Method used by the model as the criterion for training.
    Cross entropy loss used as the loss function
    :param outputs: Predicted labels by the model
    :param pictures: Actual labeled images from the dataset
    :return: The sum of the cross entropy loss function.
    """
    losses = 0

    for i, key in enumerate(outputs):
        loss_func = nn.CrossEntropyLoss()
        labelsTensor = pictures['class'].clone().detach()
        losses += loss_func(outputs[key], labelsTensor.long().to(device))

    return losses


def training(model, device, lr_rate, epochs, train_loader):
    """
    Method used by the model for training
    :param model: The model to train
    :param device: Which device to use for computation, GPU or CPU
    :param lr_rate: The learning rate used by the optimizing function
    :param epochs: How many epochs to train the model for
    :param train_loader: The loader that provides the labeled images in batches
    :return: An array containing the losses after each epoch
    """
    num_epochs = epochs
    losses = []
    checkpoint_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        for i, pictures in enumerate(train_loader):
            images = pictures['image'].to(device)

            output = model(images)

            loss = criterion(output, pictures)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (int(n_total_steps / 1)) == 0:
                checkpoint_loss = torch.tensor(losses).mean().item()
                checkpoint_losses.append(checkpoint_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {checkpoint_loss:.4f}')

    # Snippet used to save the models for inferring during runtime.
    # torch.save({
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'loss': checkpoint_losses,
    # }, '../model/house/house_model_12.tar')

    return checkpoint_losses


# Call the method to train the model
checkpoint_losses = training(model, device, 0.0001, 12, train_loader)


def validation(model, dataloader):
    """
    Method used to validate the model after training
    :param model: The model to validate
    :param dataloader: The loader that provides the labeled images in batches
    :return: The percentage of accuracy of the model.
    """
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for pictures in dataloader:
            images = pictures['image'].to(device)
            outputs = model(images)
            labels = [pictures['class'].to(device)]

            for i, out in enumerate(outputs):
                _, predicted = torch.max(outputs[out], 1)
                n_correct += (predicted == labels[i]).sum().item()

                if i == 0:
                    n_samples += labels[i].size(0)

    acc = 100.0 * n_correct / n_samples
    print(str(acc) + "%")
    return acc


# Call the method to validate the model
validation(model, test_loader)
