# Handwritten digit recognition by CNN, Streamlit

An **interactive web app** that recognizes handwritten digits (0–9) drawn on a canvas.  
Built with a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**, and deployed with **Streamlit** for a simple browser-based interface.  

---

## Features
-  **Interactive canvas**: Draw any digit (0–9) directly in the browser.
-  **Real-time predictions**: Model predicts instantly once the "Predict" button is pressed.
-  **Custom-trained CNN**: Achieves ~99% test accuracy on MNIST.
-  **Robust preprocessing pipeline**: Converts freehand drawings into MNIST-compatible 28×28 grayscale images:
  - Converts RGBA → grayscale  
  - Thresholding + bounding-box cropping  
  - Aspect-ratio preserving resize to 20×20  
  - Center padding into 28×28  
  - Normalization (0–1 scale)

---

## 📂 Repository Structure

- mnist_cnn.h5 and mnist_cnn.keras # Pretrained CNN model (Keras)  
- mnist_debug_app.py # Streamlit app with debug outputs  
- streamlit_script.py # Final production app (clean version)  
- train_model.py # (Optional) script to train and save model  
- requirements.txt # Python dependencies  
- README.md # Project documentation  


---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### 2. Create and activate environment (recommended)
```bash
conda create -n mnist-app python=3.10 -y
conda activate mnist-app
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run streamlit_script.py
```
### 5. Sample image
<img width="902" height="737" alt="image" src="https://github.com/user-attachments/assets/9da7680a-6a47-4190-aebf-cd0f1c77b4e4" />

### 6. Acknowledgements
- MNIST dataset
- Tensorflow/Keras
- Streamlit
