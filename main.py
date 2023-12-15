# Project Description
# Hand tracking for sign language using Classification ML

from train_classifier import TrainModel
from live_testing_classifier import LiveTestingModel
from extract_features import featureExtractionCoordinates
print("================================================================================")
print("ASL Recognition using MediaPipe Landmarks")
print("--------------------------------------------------------------------------------")
print("For best results, hand signs should be about 20 inches in front of the camera")
print("This program has two different models to test")
print("To test the difference between the regular model and the extra parameter model,")
print("try signing the letters A, T, N, and M consecutively. The extra parameter model")
print("performs slightly better")
print('')
print('Enter 0 to try the regular model, or enter 1 to try the model with extra parameters (improved)')
print("================================================================================")


while True:
    model_choice = input("Enter your choice (0 or 1): ")
    if model_choice in ['0', '1']:
        model_choice = int(model_choice)
        break
    else:
        print("Invalid input. Please enter 0 or 1.")


# ----------------------------------------
# Only Needed For Feature Extraction and Training
# Uncomment if need to train data again or if training new data

# confidence = 0.8
# dataset_path_name = 'created_ASL_dataset_right_hand'
# feature_path = 'right_hand_features.pickle'
# ----------------------------------------
num_instances = 250

if model_choice == 0:
    extra_parameters = False
    num_parameters = 42
    model_path = 'regular_model.pickle'
elif model_choice == 1:
    extra_parameters = True
    num_parameters = 44
    model_path = 'extra_parameters_model.pickle'

# ----------------------------------------
# Only Needed For Feature Extraction and Training
# Uncomment if need to train data again or if training new data

# featureExtractionCoordinates(num_instances, num_parameters, confidence, dataset_path_name, feature_path,
#                              extra_parameters)
# TrainModel(feature_path, model_path)
# ----------------------------------------

LiveTestingModel(model_path, num_parameters, extra_parameters)
print("Bye!")
