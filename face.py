import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img2 = cv2.imread('sunglasses.png',-1)
img3 = cv2.imread('hat.png',-1)
small = cv2.resize(img2, (0,0), fx=0.125, fy=0.125) 
small2 = cv2.resize(img3, (0,0), fx=0.125, fy=0.125) 

x_offset = 0
y_offset = 0

x2_offset = 0
y2_offset = 0

global draw
draw = False
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
       	# draw = not draw
       	# print draw
       	global draw
       	draw = not draw


while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	img = frame
	gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.setMouseCallback('img',on_mouse,draw)

	# resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
	    if w > 50 or h > 50:
	    	small = cv2.resize(img2, (w,h))
	    	small2 = cv2.resize(img3, (w/2,h/2))    	 
	    	x_offset = x
	    	y_offset = y
	    	x2_offset = x+w/2
	    	y2_offset = y-h/2
	
	for c in range(0,3):
		if draw:
			img[y_offset:y_offset+small.shape[0], x_offset:x_offset+small.shape[1], c] =  small[:,:,c] * (small[:,:,3]/255.0) +  img[y_offset:y_offset+small.shape[0], x_offset:x_offset+small.shape[1], c] * (1.0 - small[:,:,3]/255.0)
		
		img[y2_offset:y2_offset+small2.shape[0], x2_offset:x2_offset+small2.shape[1], c] =  small2[:,:,c] * (small2[:,:,3]/255.0) +  img[y2_offset:y2_offset+small2.shape[0], x2_offset:x2_offset+small2.shape[1], c] * (1.0 - small2[:,:,3]/255.0)
	
	cv2.imshow("img", img)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		cv2.imwrite('dealwithit.png', img)
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
