import numpy as np
import cv2

from keras.models import load_model


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []



def road_lines(image):

    # Get image ready for feeding into model
    print(image.shape)
    ogimg=cv2.resize(image, (500,600))
    small_img = cv2.resize(image, (160,80 ))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]
    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    # lane_image = cv2.resize(lane_drawn, (1280, 720))
    lane_image = cv2.resize(lane_drawn, (500,600))
    lane_image = lane_image.astype(np.uint8)
    # cv2.imshow("image",lane_image)
    # cv2.waitKey(0)
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(ogimg, 1, lane_image, 1, 0)

    return result


def addText(img, radius, direction):

    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_TRIPLEX

    if (direction != 'Straight'):
        text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Radius of Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)

    cv2.putText(img, text , (50,100), font, 0.8, (0,0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text1, (50,150), font, 0.8, (0,0, 255), 2, cv2.LINE_AA)

    return img



model = load_model('full_CNN_model.h5')
from find_curve import get_curve
# Create lanes object
lanes = Lanes()

cap = cv2.VideoCapture('Test datas/IMG_5686.mp4')
count = 0
while cap.isOpened():
    ret,frame = cap.read()

    curveRad, curveDir =get_curve(frame)
    print("curve radius-->",curveRad)
    print("curvedir--->",curveDir)
    result = road_lines(frame)
    finalImg = addText(result, curveRad, curveDir)
    cv2.imshow('output image', result)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        	break
cap.release()
cv2.destroyAllWindows()
