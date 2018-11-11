from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import ast
import base64
import numpy as np
from django.http import HttpResponse, JsonResponse
from imutils import contours
import cv2
import numpy as np
from matplotlib import pyplot as plt 
from keras.models import model_from_json

# Create your views here.

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

dict={0:'a', 1:'aa', 2:'e', 3:'ee', 4:'u', 5:'oo', 6:'eh', 7:'aeh', 8:'i', 9:'oh', 10:'o'}

def max_val(f):
    new=f.tolist()
    flattened=[]
    for slist in new:
        for val in slist:
            flattened.append(val)
    return(flattened)

@api_view(["POST"])
def imageConvert(image):
    shape = ast.literal_eval(image.POST.get('shape'))
    buffer = base64.b64decode(image.POST.get('image'))
    # Reconstruct the image
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(shape)
    im = cv2.imread(image,0)
    im=cv2.resize(im,(1080,1080))
    b = cv2.GaussianBlur(im,(5,5),0)
    #ret,thresh = cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edged = cv2.Canny(b, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    closed= cv2.fastNlMeansDenoising(closed,None,10,7,21)
    cv2.namedWindow("img",0)
    cv2.resizeWindow("img",540,540)
    cv2.imshow('img',closed)
    (_,cnts, h) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = contours.sort_contours(cnts, method="top-to-bottom")[0]
    s=[]
    ss=[]
    xx=[]
    yy=[]
    for c in refCnts:
        # approximate the contour
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        xx.append(x)
        yy.append(y)
        z=cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        q1=im[y:y+h,x:x+w]
        # get the min area rect
        ret,q = cv2.threshold(q1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ss.append(q)    
        if (q.shape>(60,60)):
            #s.append(q)
            plt.imshow(q)
            r=cv2.resize(q,(54,54))
            s.append(r)
           # s.reverse()
    print(len(s))
    cv2.namedWindow("img",0)
    cv2.resizeWindow("img",540,540)
    cv2.imshow('img',im) 
    l=np.asarray(s)
    p=l.astype("float32")
    n1=len(s)
    p/=255
    z=p.reshape(n1, 54, 54,1)
    f=loaded_model.predict_classes(z)
    ff=loaded_model.predict(z)
    fin=f.tolist()
    output=', '.join(str(j) for j in fin)
   
    return JsonResponse(output)