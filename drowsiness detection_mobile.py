import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from os.path import dirname, join


def main(data, score):

    # Model names, load them
    model_name = join(dirname(__file__), "model_new_data3_intra_db.tflite")
    haar_cascade_lefteye = join(dirname(__file__), "haarcascade_lefteye_2splits.xml")
    haar_cascade_righteye = join(dirname(__file__), "haarcascade_righteye_2splits.xml")

    leye = cv2.CascadeClassifier(haar_cascade_lefteye)
    reye = cv2.CascadeClassifier(haar_cascade_righteye)

    # Interpreter (tensorflow)
    interpreter = tflite.Interpreter(model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    #Cast the frame to cv2
    frame = cv2.imdecode(np.asarray(data, np.int8), cv2.IMREAD_UNCHANGED)

    #Parameters, initialize to 99 in case there is no detection
    rpred = [99]
    lpred = [99]

    #Apply the Haar cascadas

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)



    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (32, 32))
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGRA2RGB)
        r_eye = r_eye.reshape(32, 32, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        # Prediction
        r_eye = r_eye.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], r_eye)
        interpreter.invoke()
        rpred = np.round(interpreter.get_tensor(output_details[0]['index']))


    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (32, 32))
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGRA2RGB)
        l_eye = l_eye.reshape(32, 32, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        # Prediction
        l_eye = l_eye.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], l_eye)
        interpreter.invoke()
        lpred = np.round(interpreter.get_tensor(output_details[0]['index']))


    if (rpred[0] == 0 or lpred[0] == 0):
       score = score + 1
    if (rpred[0] == 1 or lpred[0] == 1):
       score = score - 1

    if (score < 0):
       score = 0


    return score



