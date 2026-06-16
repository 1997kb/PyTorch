# 1. Data Tranformation
```python
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
```

## transform.Compose is used to chain multiple image tranformation techniques into a sequential pipeline
## .ToTensor()  is used to apply scaling i.e convert [0,255] down to normalized float range of [0.0, 1.0]
## it also changes shape layout from HWC into pytorch' expected format CHW(channel,height,width) format.
## .Normalize((mean),(standardDeviation))


# 2.Data Loading
```python
   train_dataset = torchvision.datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=transform
)
```
## loading training dataset with train = True and using the defined the transformation

# 3 Batch Data Loading
```python
  train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
  test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)
```
## batch_size = 64 means dividing the 60000 images into smaller chunks of 64 images per batch . the model updates its weights after every mini-batch. For testing the batch_size = 1000 . this is a much larger size as testing doesnt require backwad propagation, it consumes less memory.

```python
  distances = [
    [1.0],  # Sample 1: Distance of first trip
    [2.0],  # Sample 2: Distance of second trip
    [3.0],  # Sample 3: Distance of third trip
    [4.0]   # Sample 4: Distance of fourth trip
]
```
## Batch Size (Outer Bracket Count): 4. You are passing 4 examples to the model at once.

## Features (Inner Bracket Count): 1. Each example contains only one piece of information (a single distance number).

## PyTorch Shape: [4, 1]

```python
  inputs = [
    [0.0, 0.13, ..., 0.0],  # Sample 1: Image 1
    [0.0, 0.0,  ..., 0.2],  # Sample 2: Image 2
    ...                     # (Imagine 60 more rows in here)
    [0.4, 0.3,  ..., 0.0]   # Sample 64: Image 64
]
```

## batch size 64 i.e passing 64 images at once
## each batch 