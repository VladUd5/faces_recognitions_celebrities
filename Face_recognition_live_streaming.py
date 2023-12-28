import face_recognition
import pickle
import cv2
import os
 
# xml file containing haarcascade file from cv2
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
cascPathface = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

# load the faces and embeddings
data = pickle.loads(open('face_enc', "rb").read())
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    #BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=3,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # take embeddings image
    encodings = face_recognition.face_encodings(rgb)
    names = []

    # loop over the facial embeddings incase

    for encoding in encodings:
        #Comparing the vectors of two images
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                # Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            # set name which has highest count
            name = max(counts, key=counts.get)
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
