import numpy as np
import argparse
import json
from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
  ag = argparse.ArgumentParser()
  ag.add_argument('input', help='Single input image path')
  ag.add_argument('project', help='Project configuration (PyTorch file .pt)')
  ag.add_argument('--top_k', metavar='N', type=int, default=5, help='Number of most likely classes')
  ag.add_argument('--category_names', default='cat_to_name.json', help='Category names json')
  ag.add_argument('--gpu', default=False, help='Use gpu instead of cpu')
  return ag.parse_args()

def get_device(gpu):
  if torch.cuda.is_available() and gpu:
    return "cuda"
  else:
    return "cpu"

def process_image(image):
  mean, deviation = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  rotation, crop_size, resize = 30, 224, 255

  im = Image.open(image)
  im = im.resize((resize, resize))
  
  width, height = im.size
  left = (width - crop_size)/2 
  top = (height - crop_size)/2
  im = im.crop((left, top, left + crop_size, top + crop_size))
  
  np_image = np.array(im)
  np_image = np_image / resize
  np_image -= np.array(mean)
  np_image /= np.array(deviation)
  
  np_image = np_image.transpose((2, 0, 1))
  
  return torch.from_numpy(np_image)

def predict(image_path, model, device, topk=5):
    model.to(device)

    with torch.no_grad():
        model.eval()
        np_img = process_image(image_path).unsqueeze_(0).float()
        logps = model.forward(np_img.to(device))
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        return top_p.cpu().numpy().ravel(), top_class.cpu().numpy().ravel()

def load(file_path):
  prev_state = torch.load(file_path)

  curr_model = getattr(models, prev_state.get('arch'))
  curr_model = curr_model(pretrained=True)
  curr_model.classifier = prev_state.get('classifier')
  curr_model.load_state_dict(prev_state.get('optimizer_state'))
  curr_model.class_to_idx = prev_state.get('mapping')
  return curr_model

def main():
  in_arg = get_input_args()

  with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

  torch_model = load(in_arg.project)

  device = get_device(in_arg.gpu)

  probs, classes = predict(in_arg.input, torch_model, device, in_arg.top_k)

  for ii in range(len(classes)):
    prob = probs[ii] * 100
    prob = round(prob,2)
    print(str(ii + 1) + 'º - ' + cat_to_name[str(classes[ii])] + ' with acurracy of ' + str(prob))
    

if __name__ == "__main__":
  main()