# White Lotus Facial Recognition

This project uses a Convolutional Neural Network (CNN) with transfer learning to perform facial recognition on characters from HBO's *The White Lotus*. It identifies characters from across the show's three seasons using the VGGFace deep learning model as a base.

## About

- Identifies 6 characters across 3 seasons of *The White Lotus*
- Uses a frozen VGGFace base model with custom top layers
- Dataset consists of manually cropped face images of each character
- Achieves high accuracy on test set and performs well on unseen stills

## Model Architecture
- Base model: VGGFace (ResNet50)
- Top layers: Global average pooling, dense layers, dropout, softmax
- Trained with early stopping and learning rate scheduling

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Resizing

model = Sequential()
model.add(Resizing(224, 224))  # ResNet50 expects 224x224 input
model.add(base_model)          # Pretrained VGGFace base
model.add(Flatten())           # Flatten 4D output to 1D
model.add(Dense(128, activation='relu'))  # Hidden layer to learn character-specific features
model.add(Dense(len(labels), activation='softmax'))  # Output layer for classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
Notes:
	
 	•	Flatten() transforms the feature maps from the base model into a single vector for classification.
	•	Dense(128, relu) learns distinctive features for each character in the dataset.
	•	Dense(softmax) outputs probabilities for each class (character).
	•	Adam is an adaptive optimizer well-suited for image data.
	•	sparse_categorical_crossentropy is appropriate for integer-labeled data.
 
## Why VGGFace?
VGGFace is a pretrained model specifically designed for facial recognition. It extracts high-level facial features such as spacing between the eyes, nose shape, jawline, etc., that are consistent across human faces. By freezing the weights of the base model, we retain this powerful feature extractor and avoid overfitting to a small custom dataset.


## Requirements:
	•	TensorFlow
	•	NumPy
	•	OpenCV
	•	Matplotlib
	•	Git LFS (for downloading vggface.h5)

## Notes
	•	This model was trained on hand-labeled and manually cropped images.
	•	It works best on stills or well-lit portraits similar to the training data.

## How to Use

1. Clone the repository:

git clone https://github.com/shainalav/White_Lotus_Facial_Recognition.git
cd White_Lotus_Facial_Recognition

2. Ensure you have Python 3.8+ and install dependencies:
pip install -r requirements.txt

3. Run predictions on a new image:
python predict.py --image path/to/image.jpg

## Acknowledgments
	•	VGGFace for the pretrained facial recognition backbone
	•	HBO’s The White Lotus for character image inspiration
 
