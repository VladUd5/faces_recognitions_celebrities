import face_recognition
import pickle
import cv2
import os
import tarfile
# Put path to the directory containing the images
# The code is initializing two empty lists, `knownEncodings` and `knownNames`.
# This code is performing face recognition and encoding on a set of images and saving the encodings
# and corresponding names in a database file using pickle.
imagePaths = os.path.join('data', 'Faces_data.tar')
knownEncodings = []
knownNames = []
tar = tarfile.open(imagePaths, "r:tar")
for tarinfo in tar:   
    if tarinfo.name[-4:]!=".jpg":
        print('Invalid format file. Only jpg {}'.format(tarinfo.name))
    else:
        name = tarinfo.name.split('/')[-2]
        tar.extract(tarinfo.name)
        image = cv2.imread(tarinfo.name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,model='hog')
    # creating vectors for each face
        encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
# save encodings and names in a database
data = {"encodings": knownEncodings, "names": knownNames}

# use picle to save encodings and names
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()
