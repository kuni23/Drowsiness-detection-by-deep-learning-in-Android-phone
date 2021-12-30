# Drowsiness-detection-by-deep-learning-in-Android-phone
This repository includes a drowsiness detection system realized by ResNet-50 neural network architecture and embedded into Android environment. Haar cascades are used for eyes extraction and deep learning for final classification. The Python module is integrated into the application by Chaquopy. Datasets used for training: MRL Eye Dataset , Kaggle Drowsiness Dataset, Closed Eyes In The Wild (CEW).

coverter.py: used to convert the tensorflow models into lite models.

model_x.py files: training module of different model considerations

drowsiness_detection_mobile.py: Python module integrated to the app

activity_main.xml : main layout of the app (frontend)

MainActivity.java: backend of app, which calls the Python module
