# Detection of Diseases in Tomato Leaves Using Convolutional Neural Networks (CNN)

## Project Overview
This project utilizes a **Convolutional Neural Network (CNN)** to detect diseases in tomato leaves, with a primary focus on identifying **bacterial spots**. The model is designed to help farmers and agricultural experts quickly diagnose and address diseases, promoting healthier crops and improving yield.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction
Diseases in tomato plants are a significant challenge for agriculture, reducing yield and affecting food quality. Identifying diseases early helps in taking prompt measures to minimize crop loss. This project automates disease detection by analyzing leaf images, providing a valuable tool for the agriculture industry.

---

## Dataset
- **Source**: PlantVillage Dataset (or specify the actual source used).
- **Content**: The dataset consists of labeled images of tomato leaves in various health conditions, including diseased and healthy categories.
- **Classes**: The model detects diseases such as:
  - Bacterial spots
  - Additional conditions (if more classes are included).

---

## Model Architecture
The model employs a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras for high accuracy in image classification.

- **Layers**:
  - Convolutional layers with max pooling for feature extraction.
  - Dense layers for classification.
- **Activation Function**:
  - ReLU activation in hidden layers.
  - Softmax in the output layer for disease classification.
- **Loss Function**: Categorical Cross-Entropy, suitable for multi-class classification.
- **Optimizer**: Adam optimizer for efficient gradient descent.

---

## Installation
### Prerequisites
- **Python 3.x**
- **TensorFlow**
- **Jupyter Notebook** (optional, if running the project in a notebook environment)

Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

---

## Usage
### Clone the Repository
```bash
git clone https://github.com/sarwat-mumtaz/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection
```

### Run the Jupyter Notebook
Open `disease_detection_in_tomato_leaves_using_CNN.ipynb` in Jupyter Notebook, and execute the cells to:
- Preprocess the data
- Train the model
- Evaluate its performance

### Inference
To test the model on new images:
1. Use an image of a tomato leaf.
2. Load it as an input.
3. Run the prediction cells to classify it as **healthy** or **diseased**.

---

## Results
The model achieves high accuracy on the test set, effectively distinguishing between healthy and diseased leaves.

- **Accuracy**: Provide the final accuracy score, e.g., 95%.
- **Precision & Recall**: (Optional) Include these metrics if available.
- **Sample Outputs**: Display a few sample images with the model’s predictions to demonstrate accuracy.

---

## Future Improvements
Potential enhancements include:
- **Larger Dataset**: Training on a larger dataset to improve robustness.
- **Additional Categories**: Adding more disease categories to make the model comprehensive.
- **Data Augmentation**: Implementing techniques to further enhance accuracy.
- **Deployment**: Deploying the model as a web or mobile app for easier access.

---

## Contributing
Contributions are welcome! Feel free to fork this repository, submit issues, or create pull requests with improvements.

---

## License
This project is licensed under the [MIT License](LICENSE).

