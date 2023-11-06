# Monitoring Banana Freshness with GoogLeNet

This project applies the GoogLeNet prebuilt model, available in the KERAS package as InceptionV3, to predict the freshness of bananas. By utilizing the concept of transfer learning, we can concatenate a custom model designed to assess banana freshness with the GoogLeNet model.

## Dataset

The dataset comprises three types of images representing different stages of banana ripeness: RIPE, GREEN, and OVER RIPE. These images are organized within the `Dataset/train` directory, with each category in its respective folder.


## Model Concatenation

The custom banana freshness model is concatenated with the GoogLeNet model using the dataset mentioned above. The process is documented in the code, with red comments explaining the use of the GoogLeNet model.

## Running the Project

To run the project, double-click on the `run.bat` file. This will launch the following interface:

![Project Interface](Dataset\Picture4.png)

*Above: Initial project interface.*

### Uploading the Dataset

Click on the `Upload Banana Dataset` button to upload the dataset and proceed to the following interface:

After selecting and uploading the `Dataset` folder, click on `Select Folder` to load the dataset.

### Model Generation and Evaluation

With the dataset loaded, click on `Generate & Load GoogLeNet Model` to build the model based on the loaded dataset and calculate its accuracy and loss.
- Model generated with an accuracy of 82%.*

The GoogLeNet layers can be viewed in the console:

![GoogLeNet Layers](Dataset\Picture8.png)

*Above: Console output showing GoogLeNet layers.*

### Accuracy & Loss Graph

Click on the `Accuracy & Loss Graph` button to visualize the model's performance over epochs.

## Predicting Banana Freshness

To predict the freshness of a new banana image:

1. Click on `Upload Banana Test Image & Monitor Change`.
2. Upload a test image, and the model will predict its freshness.

### Prediction Results

The model will output the predicted freshness category for the uploaded image.Continue uploading images to test the model's predictions.

## Conclusion

This model provides a convenient way to monitor the change in banana freshness. By uploading images of bananas, the model can predict whether they are RIPE, GREEN, or OVER RIPE with a reasonable degree of accuracy.

