import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
  ag = argparse.ArgumentParser()
  ag.add_argument('data_dir', help='Path to folder with train, validation and test images')
  ag.add_argument('--arch', default="vgg11", help='PyTorch base model. Possible values: vgg11|vgg13|vgg16|vgg19')
  ag.add_argument('--learning_rate', default=0.01, help='Learning rate')
  ag.add_argument('--hidden_units', default=512, help='Number of hidden units')
  ag.add_argument('--epochs', default=1, help='Number of epochs')
  ag.add_argument('--gpu', default=False, help='Use gpu instead of cpu')
  ag.add_argument('--save_dir', default=os.getcwd(), help='Path to save current state')
  ag.add_argument('--batch_size', default=32, help='Batch size for DataLoader')
  return ag.parse_args()

def build_directories(data_dir):
  return data_dir + '/train', data_dir + '/valid', data_dir + '/test'

def check_arch(arch):
  possible_arch = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
  if arch not in possible_arch:
    raise Exception('Invalid arch. Plese define one of those archs: ' + str(possible_arch))

def get_device(gpu):
  if torch.cuda.is_available() and gpu:
    return "cuda"
  else:
    return "cpu"
  
def build_classifier(arch, hidden_units, n_classes):
  check_arch(arch)

  torch_model = getattr(models, arch)
  torch_model = torch_model(pretrained=True)

  for param in torch_model.parameters():
      param.requires_grad = False

  classifier = nn.Sequential(
    OrderedDict([(
      'fc1', nn.Linear(25088, hidden_units)),
      ('relu1', nn.ReLU()),
      ('dp1', nn.Dropout(p=0.5)),
      ('fc2', nn.Linear(hidden_units, n_classes)),
      ('output', nn.LogSoftmax(dim=1))
      ]))

  torch_model.classifier = classifier

  return torch_model

def validation(model, loader, crit, device):   
  loader_size = len(loader)
    
  validation_loss = 0
  accuracy = 0

  with torch.no_grad():
    for inputs, labels in loader:
      inputs, labels = inputs.to(device), labels.to(device)
      logps = model.forward(inputs)
        
      validation_loss += crit(logps, labels).item()

      ps = torch.exp(logps)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)
      accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
  return validation_loss/loader_size, accuracy/loader_size

def train(model, trainingloader, validationloader, epochs, learning_rate, device):  
  momentum = 0.5

  criterion = nn.NLLLoss()

  optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum = momentum)

  steps = 0
  print_every = 20

  model.to(device)
  
  for epoch in range(epochs):
    running_loss = 0

    for inputs, labels in trainingloader:
      steps += 1

      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()

      logps = model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if steps % print_every == 0:
          model.eval()

          v_loss, v_acc = validation(model, validationloader, criterion, device)

          model.train()  

          print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Validation loss: {v_loss:.3f}.. "
                f"Validation accuracy: {v_acc:.3f}")

          running_loss = 0

def save(model, mapping, base_dir, arch):
  torch.save({
    'classifier': model.classifier,
    'optimizer_state': model.state_dict(),
    'mapping': mapping,
    'arch': arch,
    }, base_dir + '/project.pt')

def test(model, loader, device):
    test_size = 0
    n_pred = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            tensor, indices = torch.max(logps.data, 1)
            test_size += labels.size(0)
            n_pred += (indices == labels).sum().item()
    return (n_pred / test_size) * 100

def main():
  in_arg = get_input_args()

  train_dir, valid_dir, test_dir = build_directories(in_arg.data_dir)

  mean, deviation = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  rotation, crop_size, resize = 30, 224, 255

  training_transforms = [transforms.RandomRotation(rotation),
                        transforms.RandomResizedCrop(crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, deviation)]

  check_transforms = [transforms.Resize(resize),
                      transforms.CenterCrop(crop_size),
                      transforms.ToTensor(),
                      transforms.Normalize(mean, deviation)]

  training_data_trans = transforms.Compose(training_transforms)
  training_img_dataset = datasets.ImageFolder(train_dir, transform=training_data_trans)
  trainingloader = torch.utils.data.DataLoader(training_img_dataset, batch_size=in_arg.batch_size, shuffle=True)

  validation_data_trans = transforms.Compose(check_transforms)
  validation_img_dataset = datasets.ImageFolder(valid_dir, transform=validation_data_trans)
  validationloader = torch.utils.data.DataLoader(validation_img_dataset, batch_size=in_arg.batch_size, shuffle=True)

  test_data_trans = transforms.Compose(check_transforms)
  test_img_dataset = datasets.ImageFolder(test_dir, transform=test_data_trans)
  testloader = torch.utils.data.DataLoader(test_img_dataset, batch_size=in_arg.batch_size, shuffle=True)

  with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
  n_classes = len(cat_to_name)

  device = get_device(in_arg.gpu)

  print('Using device: ' + device)

  torch_model = build_classifier(in_arg.arch, in_arg.hidden_units, n_classes)

  print('Classifier definition')
  print(torch_model)

  print('Training model')

  train(torch_model, trainingloader, validationloader, in_arg.epochs, in_arg.learning_rate, device)  

  print('Checking model accuracy')
  acc = test(torch_model, testloader, device)

  print('Model accuracy:' + str(acc))

  save(torch_model, training_img_dataset.class_to_idx, in_arg.save_dir, in_arg.arch)

if __name__ == "__main__":
  main()