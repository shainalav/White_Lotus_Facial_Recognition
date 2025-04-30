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

## Requirements:
	•	TensorFlow
	•	NumPy
	•	OpenCV
	•	Matplotlib
	•	Git LFS (for downloading vggface.h5)

## Notes
	•	This model was trained on hand-labeled and manually cropped images.
	•	It works best on stills or well-lit portraits similar to the training data.

## Acknowledgments
	•	VGGFace for the pretrained facial recognition backbone
	•	HBO’s The White Lotus for character image inspiration
 
## How to Use

1. Clone the repository:

```bash
git clone https://github.com/shainalav/White_Lotus_Facial_Recognition.git
cd White_Lotus_Facial_Recognition

2. Ensure you have Python 3.8+ and install dependencies:
pip install -r requirements.txt

3. Run predictions on a new image:
python predict.py --image path/to/image.jpg

