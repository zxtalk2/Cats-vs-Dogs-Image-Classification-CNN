# 🐾 Cats vs Dogs Image Classification

A custom 4-block Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras to classify images of cats and dogs. Instead of using transfer learning (like MobileNet or VGG), I designed and tuned this architecture manually to achieve solid performance on fully unseen data.

🌍 **Live Demo:** [Test your own images here!](https://huggingface.co/spaces/zxtalk/cats-vs-dogs-classification)

---

## 📊 Project Results
To make sure the model actually learns patterns rather than just memorizing data, I split the dataset to keep the final test data completely isolated during training and hyperparameter tuning.

*   **Validation Accuracy:** 92.96% (Used during training)
*   **Test Accuracy:** 88.34% (On completely unseen data)
*   **Test Loss:** 0.7877

---

## 🛠️ Tech Stack
*   **Deep Learning Framework:** TensorFlow 2.x & Keras
*   **Deployment:** Hugging Face Spaces & Gradio
*   **Core Libraries:** NumPy, Matplotlib, Pillow, Scikit-Learn
*   **Training Hardware:** Google Colab (T4 GPU)

---

## 💡 Key Design & Optimization Choices

Here is what I implemented to make the model efficient and prevent overfitting:

*   **Data Pipeline (`tf.data`):** Used `.cache()` and `.prefetch()` to speed up training, ensuring the GPU doesn't sit idle while data is loading.
*   **Regularization Against Overfitting:** Added L2 regularization (0.001) inside the conv layers, along with a 40% Dropout layer right before the final classification.
*   **Global Average Pooling (GAP):** Swapped out the traditional heavy `Flatten` layer for `GlobalAveragePooling2D`. This drastically dropped the total number of trainable parameters and helped control validation variance.
*   **Dynamic Learning Rate & Early Stopping:** Integrated Keras callbacks to automatically lower the learning rate whenever validation loss started to plateau, and stopped training early to restore the absolute best weights.
*   **Deployment:** Converted the backend setup to map smoothly with Gradio and deployed it as a live web app on Hugging Face Spaces.

---

## 📐 Model Architecture

The entire network structure is built sequentially as follows:

1.  **Input:** `(150, 150, 3)`
2.  **Block 1:** `Conv2D(64, (3,3))` + `BatchNormalization` + `MaxPooling2D`
3.  **Block 2:** `Conv2D(128, (3,3))` + `BatchNormalization` + `MaxPooling2D`
4.  **Block 3:** `Conv2D(256, (3,3))` + `BatchNormalization` + `MaxPooling2D`
5.  **Block 4:** `Conv2D(512, (3,3))` + `BatchNormalization` + `MaxPooling2D`
6.  **Pooling:** `GlobalAveragePooling2D()`
7.  **Dense Head:** `Dense(256, Activation='relu')` -> `Dropout(0.4)` -> `Dense(1, Activation='sigmoid')`
