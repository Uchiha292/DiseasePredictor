# DiseasePredictor

Predicts Top 3 Diseases Based on Symptoms<br />
This project applies deep learning to predict medical conditions based on patient symptoms. Using a dataset of 132 symptoms mapped to 41 diseases, we train a neural network classifier that assists in the early diagnosis of diseases, potentially helping healthcare professionals make faster decisions.

<h1>Project Overview<br /></h1>
Domain: Medical Diagnosis<br />
Techniques Used: Supervised Machine Learning, Deep Neural Networks<br />
Model Type: Multilayer Perceptron (Keras Sequential API)<br />
Dataset: 132 binary symptoms → 1 out of 41 diseases

<h1>Model Training</h1>
Neural Network Architecture<br />
• Input layer: 132 neurons (symptoms)<br />
• Hidden layers: 64 → 32 neurons<br />
• Output layer: n_classes neurons<br />
Training Configuration<br />
• 20 epochs<br />
• Batch size of 32

<h1>How To Run</h1>
• Clone the repo<br />
• Install requirements: pip install gradio numpy tensorflow<br />
• python App.py<br />
