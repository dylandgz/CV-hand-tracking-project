
# Sign Language Detector - CV-hand-tracking-project 

This project is a sign language detector for the American Sign Language (ASL) alphabet.

## Dependencies

To run this project, install the following dependencies:

- Python 3.10
- OpenCV (Open Source Computer Vision Library)
- MediaPipe (Framework for building multimodal applied machine learning pipelines)
- Scikit-learn (Machine learning library for Python)

```bash
pip install opencv-python
pip install mediapipe
pip install scikit-learn
```

## Dataset

The initial dataset used for this project can be found on Kaggle:

- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Modifications to the dataset:

- Removed the `nothing` folder.
- Removed the `space` folder.
- Removed the `j` and `z` letters due to the requirement for movement.

### Custom Dataset:

- Developed a custom dataset to train the models (not included in the GitHub repository).

## Models

There are two different models available for detection:

1. A basic model using MediaPipe and landmarks to detect the signs.
2. An advanced model that includes extra parameters derived from the distances between landmarks to better differentiate similar hand signs.

For more information on these models, follow the instructions given after running the code in the terminal.

## Running the Project

Navigate to the folder containing the project files in the terminal and execute the main script:

```bash
python main.py
```

Press `q` to exit detection.

## Additional Notes

- It is recommended to run this project in a virtual environment with python 3.10
- Further instructions and model details are provided in the terminal after running the main script.
