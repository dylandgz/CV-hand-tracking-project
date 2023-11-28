# Sign Language Detector - CV-hand-tracking-project 

This project is a sign language detector for the American sign language alphabet.(This is still a work in progress)

## Dependencies

To run this project, you need to install the following dependencies:

- Python 3.10
- OpenCV (Open Source Computer Vision Library)
- MediaPipe (Framework for building multimodal applied machine learning pipelines)
- Scikit-learn (Machine learning library for Python)

## Dataset

The dataset used for this project can be found on Kaggle:

- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Modifications to the dataset:

- Removed the `nothing` folder
- Removed `space` folder
- Removed `j` and `z` letters due to the requirement for movement

## Running the Project

Navigate to the folder containing the project files in the terminal and run the main script:

```bash
python main.py
```

press q to exit detection
