import tensorflow as tf
import numpy as np
import cv2
from IPython.display import display
from sklearn.metrics import classification_report

# importing model
model = tf.keras.models.load_model('faces_cropped.keras')
#initializing face detection
facec=cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')

print('The camera usually takes ~20/30 secs to initiate.')
# making real time predictions
# Initialize the webcam
cap = cv2.VideoCapture(0) 
font= cv2.FONT_HERSHEY_SIMPLEX
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read and display frames from the webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    #converting image to grey
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img=np.expand_dims(img,-1)
    size=(90,90)
    face=facec.detectMultiScale(img,scaleFactor=1.1,minNeighbors= 9)
    # trying in case no face is detected, this is the best could do.
    try:
        if face[0][1] > 0:   
            for (x,y,w,h) in face:
                cropped=tf.image.crop_to_bounding_box(img,y,x,h,w)
                image=tf.image.resize(cropped,size)
    # if there is no face then resize normal image
    except:
        image=tf.image.resize(img,size)
        x=100
        y=100
        h=0
        w=0
    
    image=np.array(image/255)
    
    #since it's greyscale and therefore doesn't have a 3 after the size,
    # (90,108,3) for shape in rgb it needs a one (90,108,1) for grey scale
    # since it's a single img it still needs a batch size of one to be the right shape
    image=np.expand_dims(image,0)
    
    # have to re-add the last dim 
    image=np.expand_dims(image,-1)
    try:
        pred=model.predict(image,verbose=0)
    except:
        sex='na'
        res='na'
    pred=np.round(pred)
   
    if pred[0]==1:
        res='smiling'
    elif pred[0]==0 and pred[1]==0:
        res='try smiling'
        sex='woman'
    else:
        res='try smiling'
        sex='man'

    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(frame, res,(x,y+h), font, 1, (255, 255, 0), 2)
    cv2.putText(frame, sex,(x,y), font, 1, (255, 255, 0), 2)
    cv2.putText(frame,'To quit press q',(200,400), font, 1, (0, 0,255), 2)
    cv2.imshow('Webcam', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows

cap.release()
cv2.destroyAllWindows()