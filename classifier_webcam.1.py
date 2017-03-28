#!/usr/bin/env python2
#
# Example to run classifier on webcam stream.
# Brandon Amos & Vijayenthiran
# 2016/06/21
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Contrib: Vijayenthiran
# This example file shows to run a classifier on webcam stream. You need to
# run the classifier.py to generate classifier with your own dataset.
# To run this file from the openface home dir:
# ./demo/classifier_webcam.py <path-to-your-classifier>


import time

start = time.time()

import argparse
import cv2
import os
import pickle
import imageio

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
import skvideo.io

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    pts = []
    alignedFaces = []
    for box in bb:
        pt = [(box.left(), box.bottom()), (box.right(), box.top())]
        pts.append(pt);
        landmarks = align.findLandmarks(rgbImg, box)
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarks=landmarks,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    # print reps
    return (reps, pts)


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    reps, pts = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences, pts)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    
    # START IMAGEIO to record output frames
    imageio_read = imageio.get_reader('/home/huuson235/Videos/video/1.mp4')
    fps = imageio_read.get_meta_data()['fps']
    imageio_write = imageio.get_writer('/home/huuson235/Videos/video/_1.mp4', fps=fps)

    video_capture = cv2.VideoCapture('/home/huuson235/Videos/video/1.mp4') # INPUT VIDEO
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    # video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)


    confidenceList = []

    # VIDEO_WRITER FAIL FAIL FAIL FAIL FAIL FAIL FAIL FAIL FAIL FAIL
    # video_writer = cv2.VideoWriter()
    # video_writer.open(
    #     "outputvideo.mp4",
    #     int(video_capture.get(cv2.cv.CV_CAP_PROP_FOURCC)),
    #     int(video_capture.get(cv2.cv.CV_CAP_PROP_FPS)),
    #     (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) )),
    #     True)

    while True:
        ret, frame = video_capture.read()
        persons, confidences,pts= infer(frame, args)
        print "Time: "+ time.strftime('%Hh %Mm %Ss', time.localtime()) + " P: " + str(persons) + " C: " + str(confidences)
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        for i, c in enumerate(confidences):
            if c <= args.threshold:  # 0.5 is kept as threshold for known face.
                persons[i] = ""
                # person[i] = "_unknown"

        # Print the person name and conf value on the frame
        # cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
        #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(frame, "Time: {}".format(time.strftime('%Hh %Mm %Ss', time.localtime())), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        count = 0
        for pt1, pt2 in pts:
            name =''
            if (len(persons) != 0):
                name = persons[count]

            cv2.rectangle(frame, pt1, pt2, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, name, (pt1[0], pt2[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,0,0), thickness=2)
            count=count+1
        
        # cv2.imshow('', frame)
        imageio_write.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:,:,:])
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    imageio_write.close()
    video_capture.release()
    cv2.destroyAllWindows()
