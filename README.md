# Neural Fashion Analyzer (Fashion-MNIST Classification)

## 1. Problem Description
This project focuses on automated fashion item classification using a Multilayer Perceptron (MLP) built with TensorFlow and Keras. The objective is to train a deep learning model to correctly classify grayscale images of clothing items into one of 10 distinct categories. This project serves as a practical application of Deep Learning and fulfills the core requirements for the Neural Networks course.

## 2. Dataset Information
The project utilizes the **Fashion-MNIST** dataset.
* **Dataset Link:** [Fashion-MNIST on GitHub](https://github.com/zalandoresearch/fashion-mnist)
* **Total Samples:** 70,000 grayscale images (28x28 pixels).
* **Data Split:** * Training: 55,000 samples
  * Validation: 5,000 samples
  * Testing: 10,000 samples
* **Target Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## 3. Data Preprocessing
To prepare the data for the MLP, the following preprocessing steps were applied:
* **Normalization:** Pixel values were converted to `float32` and scaled by dividing by 255.0 to ensure all values fall strictly between 0 and 1.
* **Partitioning:** The data was explicitly split to monitor validation accuracy during training.

## 4. Model Architectures & Experimentation
In accordance with the project criteria, two distinct experiments were conducted by varying the optimizer and applying regularization techniques.

### Experiment 1: Baseline MLP
* **Architecture:** Flatten Layer -> Dense (300 neurons, ReLU) -> Dense (100 neurons, ReLU) -> Dense (10 neurons, Softmax).
* **Optimizer:** Stochastic Gradient Descent (`SGD`).
* **Loss Function:** `sparse_categorical_crossentropy`.

### Experiment 2: Enhanced MLP (Regularization + Advanced Optimization)
* **Architecture:** Flatten Layer -> Dense (300 neurons, ReLU) -> **Dropout (0.3)** -> Dense (100 neurons, ReLU) -> **Dropout (0.3)** -> Dense (10 neurons, Softmax).
* **Optimizer:** Adaptive Moment Estimation (`Adam`).
* **Purpose:** The `Dropout` technique was introduced to prevent overfitting, while the `Adam` optimizer was used to accelerate convergence and improve overall accuracy.

## 5. Results & Performance Comparison
Both models were evaluated on the 10,000 testing samples. The integration of Adam and Dropout showed a noticeable improvement in the model's ability to generalize.

| Experimental Configuration | Final Test Accuracy | Final Test Loss |
| :--- | :---: | :---: |
| **Experiment 1 (Baseline SGD)** | **87.52%** | **0.3555** |
| **Experiment 2 (Adam + Dropout)** | **89.20%** | **0.3210** |

*(Note: The accuracy and loss for Experiment 2 might slightly vary upon retraining).*

## 6. Visualizations
The Jupyter notebook generates comprehensive diagnostic plots (Accuracy Curves and Loss Curves) tracking training versus validation behavior across all epochs for both experiments. 

## 7. Instructions for Running the Project
Follow these steps to run the complete code:
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/Nour-682005/neural-fashion-analyzer.git](https://github.com/Nour-682005/neural-fashion-analyzer.git)
