# Tissue Classification using Convolutional Neural Networks

## Overview
This project compares multiple deep learning models for medical image classification using the PathMNIST dataset.

The goal is to classify histopathology images into 9 tissue classes under the same training conditions.

---

## Models
- Simple CNN (baseline)
- ResNet18 (scratch)
- ResNet18 (pretrained)
- ResNet50

---


## Run the Project

```bash
pip install -r requirements.txt
python main.py
```

---

## Dataset
- PathMNIST (MedMNIST)
- 9 tissue classes in greyscale
- Image split:
	- Training set: 89,996 images
	- Validation set: 10,004 images
	- Test set: 7,180 images

---


## GitHub Pages: 
https://tjc234.github.io/cs499-project/


---


## Author
Tyler Chapp  
CS499 – Computer Vision in Healthcare


---

## License
This project is licensed under the MIT License – see the LICENSE file for details.