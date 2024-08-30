import cv2
import numpy as np
import time
import main as s

def emotion_detector(img_array):
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    time_init = time.time()
    
    test_image = cv2.resize(img_array, (256,256))
    im = np.float32(test_image)
    img_array = np.expand_dims(im, axis=0)
    time_elapsed_preprocess = time.time() - time_init

    onnx_pred = s.m_q.run(['output'], {"input": img_array})
    time_elapsed = time.time() - time_init

    emotion = ""
    if np.argmax(onnx_pred[0][0]) == 0:
        emotion = "Angry"
        
    elif np.argmax(onnx_pred[0][0]) == 1:
        emotion = "Happy"
    else:
        emotion = "Sad"

    return {"emotion": emotion, "time_elapsed": str(time_elapsed), "time_elapsed_preprocess": str(time_elapsed_preprocess)}
