# American Sign Language Detection

This repository implements an **American Sign Language (ASL)** detection system using a deep learning-based `Convolutional Neural Network (CNN)`. It is designed to classify input images of hand signs into one of 29 classes — the 26 letters of the English alphabet (`A-Z`), plus `SPACE`, `DELETE`, and `NOTHING`.

## Model Architecture

The model is a convolutional neural network (CNN) designed for multi-class image classification with 29 output classes (representing A-Z plus additional symbols). It consists of three convolutional blocks followed by fully connected layers:

- **Conv Block 1:**

  - `Conv2d` with 3 input channels and 64 output channels, kernel size 3x3, padding 1

  - Batch Normalization

  - ReLU activation

  - Max Pooling with kernel size 2x2 and stride 2

- **Conv Block 2:**

  - `Conv2d` with 64 input channels and 128 output channels, kernel size 3x3, padding 1

  - Batch Normalization

  - ReLU activation

  - Max Pooling with kernel size 2x2 and stride 2

- **Conv Block 3:**

  - `Conv2d` with 128 input channels and 256 output channels, kernel size 3x3, padding 1

  - Batch Normalization

  - ReLU activation

  - Max Pooling with kernel size 2x2 and stride 2

After the convolutional layers, the feature maps are flattened and passed through:

- A fully connected layer with 512 units and dropout (p=0.5) for regularization

- A final fully connected layer with 29 output units corresponding to the classes

The model supports optional label input to compute cross-entropy loss with label smoothing (0.1), which can be returned along with logits during training.

```text
Model(
  (conv_block1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=200704, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=29, bias=True)
)
```

## Dataset

This project utilizes the **ASL Alphabet Dataset** provided by **grassknoted** on **Kaggle**. The dataset consists of high-resolution images representing American Sign Language (ASL) alphabets.

**Key Features:**

- **Total Size:** ~1.1 GB

- **Classes:** 29 (A–Z, space, delete, nothing)

- **Samples per Class:** ~3,000 images

- **Image Dimensions:** 200x200 pixels (RGB)

**Preprocessing Steps:**
Before feeding the images into the model, the following preprocessing pipeline is applied using torchvision.transforms:

```python
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

- **Resize:** Each image is resized to 224x224 pixels.

- **ToTensor:** Converts the image to a PyTorch tensor.

- **Normalize:** Normalizes the pixel values to the range [-1, 1] using a mean and standard deviation of 0.5 for each RGB channel.

## Model Training Metrics

The following values represent the training loss recorded at the end of each of the 5 epochs, demonstrating the model's performance improvement over time:

```json
{
    "epoch_loss": [
        3.5508156776428224,
        2.5247885942459107,
        1.9314132690429688,
        1.4826382637023925,
        1.0887906670570373
    ]
}
```

> ⚠️ **Note:** The model was trained only for 5 epochs due to device constraints.

## Model Evaluation Metrics

The following results summarize the performance of the trained model, including its accuracy and average loss on the evaluation dataset:

```json
{
    "accuracy": 92.85714285714286,
    "avg_loss": 1.0683372020721436
}
```

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/ASL-Detection.git
cd ASL-Detection
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initiate DVC

```sh
dvc init
```

### Run the pipeline

```sh
dvc repro
```

The pipeline automatically launches the Flask application at:

```text
http://localhost:3000/
```

## Conclusion

This project delivers an effective solution for American Sign Language (ASL) detection using a CNN-based model, achieving high accuracy across 29 hand sign classes. With a reproducible DVC pipeline and a user-friendly Flask interface, it provides a complete workflow from training to deployment, making it easy to run, evaluate, and extend.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
