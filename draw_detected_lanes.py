import numpy as np
import cv2
# from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
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
    blanks = np.zeros_like(lanes.avg_fasdfghit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (1280, 720))
    lane_image = lane_image.astype(np.uint8)
    # cv2.imshow("image",lane_image)
    # cv2.waitKey(0)
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


model = load_model('full_CNN_model.h5')
# Create lanes object
lanes = Lanes()

cap = cv2.VideoCapture('lane.mp4')
count = 0
while cap.isOpened():
	ret,frame = cap.read()

	result = road_lines(frame)
	cv2.imshow('window-name', result)
	count = count + 1
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()


    # # Where to save the output video
    # vid_output = 'projg_vid.mp4'
    # # Location of the input video
    # clip1 = VideoFileClip("project_video.mp4")
    # # Create the clip
    # vid_clip = clip1.fl_image(road_lines)
    # vid_clip.write_videofile(vid_output, audio=False)
