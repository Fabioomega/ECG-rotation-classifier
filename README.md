# ECG Rotation Classifier
A simple rotation classifier built withe the mobilenet_small to detected whether an ECG image was rotated or not. Only works for increments of 90 degrees.
For more information and example of use see [example](https://github.com/Fabioomega/ECG-rotation-classifier/blob/main/example.py "example").

# Training and Validation
The model is provided in the `rotation_classifier` folder while the training script and loops are in the `training` folder.
The model was trained on 1k real ECG images rotated artificially during training. The validation results for that dataset are as follows:
| Metric        |  Value        |
| ------------- | ------------- |
| roc_auc       |   0.999499    |
| precision     |   0.987990    |
| recall        |   0.988374    |
| f1            |   0.988025    |
