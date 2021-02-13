import torch
from torchvision import transforms

import numpy as np
from PIL import Image

MEAN = (0.485, 0.456, 0.406)                             
STD = (0.229, 0.224, 0.225)

def load_image(img_path: str, max_size: int=400, shape: int=None) -> torch.Tensor:
  """
  img_path: path to image
  max_size: max_size for center crop, default to 400
  shape: shape for center crop if specified

  return: cropped and normalized image tensor
  """
  
  image = Image.open(img_path).convert('RGB')

  if max(image.size) > max_size:
    size = max_size
  else:
    size = max(image.size)
  
  if shape is not None:
    size = shape
  
  transform = transforms.Compose([transforms.CenterCrop(size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
  # keep the first 3 channels, alpha channel will be removed
  image = transform(image)[:3, :, :].unsqueeze(0)
  return image


def image_tensor_to_numpy(img_tensor: torch.Tensor) -> np.array:
  """
  img_tensor: image tensor of shape (batche_size, channel, height, width)

  return: image in numpy array, ready to be plotted using plt.imshow()
  """
  image = img_tensor.to('cpu').clone().detach()
  # move color channel to the last dimension
  image = image.numpy().squeeze(0).transpose(1, 2, 0)
  image = image * STD + MEAN
  image = image.clip(0, 1)
  return image


def get_features(image: torch.Tensor, model: torch.nn.Sequential, layers: dict=None)-> dict: 
  """
  image: input image tensor
  model: a model with convolution layers
  layers: optional, a dict of mapping of layer names

  return: features maps at default (or user specified) layers
  """
  if layers is None:
    # use '*_1' to get gram matrices
    # use 'conv4_2' to extract content feature
    layers = {'0': 'conv1_1', 
              '5': 'conv2_1',
              '10':'conv3_1',
              '19':'conv4_1',
              '21':'conv4_2', 
              '28':'conv5_1'}
  
  features = {}
  x = image

  # go through each layer of the model,
  # when the layer's name matches the key in predfined diction layers
  # save the feature map in features
  for name, layer in model._modules.items():
    # forward
    x = layer(x)
    if name in layers:
      features[layers[name]] = x
  
  return features


def get_gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
  """
  feature_maps: a stack of feature maps of shape 
                (batch_size=1, depth, height, width)

  return: gram_matrix, the correlation among <depth> number of feature maps
  """
  _, d, h, w = feature_maps.shape
  x = feature_maps.view(d, h * w)
  # the shape of gram matrix is (d, d)
  gram_matrix = torch.mm(x, x.t())

  return gram_matrix