import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
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




root = tk.Tk()
root.title("Lane and Curve Detection")

video_path = None
playing = False
play_thread = None
stop_frame_index = 0  # Store the frame index when stopping

canvas_width = 640  # Width of the video display canvas
canvas_height = 480  # Height of the video display canvas
c1=canvas_width-200
c2=canvas_height-100
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

def upload_video():
    global video_path, playing, play_thread
    video_path = filedialog.askopenfilename(
        initialdir="/", title="Select Video File", filetypes=(("Video Files", "*.mp4"),)
    )
    if video_path:
        playing = False
        play_video()

def play_video():
    global video_path, playing, play_thread, stop_frame_index
    playing = True
    cap = cv2.VideoCapture(video_path)

    # Set the frame index to the stopped frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, stop_frame_index)

    def update_frame():
        global playing, stop_frame_index
        while playing:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            curveRad, curveDir =get_curve(frame)
            print("curve radius-->",curveRad)
            print("curvedir--->",curveDir)
            result = road_lines(frame)
            finalImg = addText(result, curveRad, curveDir)
         

            finalImg = cv2.resize(finalImg, (c1, c2))
            image = Image.fromarray(finalImg)
            photo = ImageTk.PhotoImage(image)



             # Calculate the coordinates to center the image in the canvas
            x = (canvas_width - c1) // 2
            y = (canvas_height - c2) // 2

            canvas.create_image(x, y, image=photo, anchor=tk.NW)
            canvas.image = photo

            # Store the current frame index
            stop_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

        cap.release()

    play_thread = threading.Thread(target=update_frame)
    play_thread.start()



def pause_video():
    global playing
    playing = False

def stop_video():
    global playing, stop_frame_index
    playing = False
    stop_frame_index = 0  # Reset the stop frame index

def close_window():
    global play_thread
    if play_thread and play_thread.is_alive():
        play_thread.join()

    root.destroy()



bottom_frame = tk.Frame(root, bg="grey", height=100, padx=10, pady=10)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

upload_button = tk.Button(bottom_frame, text="Upload Video", command=upload_video, bg="light blue")
upload_button.pack(side=tk.LEFT)

play_button = tk.Button(bottom_frame, text="Play", command=play_video, bg="light green")
play_button.pack(side=tk.LEFT, padx=10)

pause_button = tk.Button(bottom_frame, text="Pause", command=pause_video, bg="light yellow")
pause_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(bottom_frame, text="Stop", command=stop_video, bg="light coral")
stop_button.pack(side=tk.LEFT, padx=10)

# Center align the buttons within the bottom_frame
bottom_frame.pack_propagate(0)
bottom_frame.update()
button_width = sum(button.winfo_reqwidth() for button in [upload_button, play_button, pause_button, stop_button])
empty_space = bottom_frame.winfo_width() - button_width
empty_label = tk.Label(bottom_frame, text='', width=empty_space, bg='grey')
empty_label.pack(side=tk.LEFT)






root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()
