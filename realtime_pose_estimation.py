### Capture real-time video stream via webcam ###
import cv2

def capture_video_from_webcam():
    # capture video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # display the resulting frame
        cv2.imshow('Video Stream', frame)

        # wait for 1ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture
    cap.release()
    cv2.destroyAllWindows()

### Application of posture recognition model ###
    
import tensorflow as tf
import numpy as np

def load_interpreter(model_path):
    """TensorFlow Lite 모델 로드"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, input_details):
    """video frame preprocessing"""
    height, width = input_details[0]['shape'][1:3]
    image = cv2.resize(image, (width, height))
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = image / 255.0
    return image

def apply_pose_estimation_model(frame, interpreter, input_details, output_details):
    """자세 인식 모델 적용"""
    input_image = preprocess_image(frame, input_details)
    
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints


def postprocess_output(frame, keypoints, threshold=0.5):
    """visualization keypoints"""
    for keypoint in keypoints[0]:
        if keypoint[2] > threshold:  # confidence score
            cv2.circle(frame, (int(keypoint[1]), int(keypoint[0])), 5, (0, 255, 0), -1)
    return frame


def main():
    model_path = '../camdata/..... .tflite'
    cap = cv2.VideoCapture(0)

    interpreter = load_interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = apply_pose_estimation_model(frame, interpreter, input_details, output_details)
        frame = postprocess_output(frame, keypoints)

        cv2.imshow('Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
