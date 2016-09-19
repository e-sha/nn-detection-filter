import numpy as np

def bbox2cornerArray(in_data):
  out_data = in_data[:, :4].copy()
  out_data[:, 2:] += out_data[:, :2]
  return out_data

def IoU(in_firstArray, in_secondArray):
  first = bbox2cornerArray(in_firstArray)
  second = bbox2cornerArray(in_secondArray)

  first = first.reshape((first.shape[0], 1, -1))
  second = second.reshape((1, second.shape[0], -1))

  br = np.minimum(first[:, :, 2:], second[:, :, 2:])
  tl = np.maximum(first[:, :, :2], second[:, :, :2])

  first = first[:, :, 2:] - first[:, :, :2]
  second = second[:, :, 2:] - second[:, :, :2]
  first = np.prod(first, axis = 2)
  second = np.prod(second, axis = 2)

  sizeArray = br - tl
  interArray = np.prod(sizeArray, axis = 2)
  mask = np.any(sizeArray < 0, axis = 2)
  interArray[mask] = 0

  return interArray / (first + second - interArray)
