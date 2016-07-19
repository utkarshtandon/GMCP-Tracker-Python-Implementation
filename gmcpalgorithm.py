from __future__ import division

'''
----------------------
This program is a Python implementation of GMCP which takes video input from the "gmcpweb.html" website 
and separately tracks each humans based on motion cost and appearance cost.
----------------------
Code is based on the paper "GMCP-Tracker: Global Multi-object Tracking Using Generalized Minimum Clique Graphs"
by Amir Roshan Zamir, Afshin Dehghan, and Mubarak Shah
----------------------
######################
Author: Utkarsh Tandon
######################
Date: 06/10/2015
######################
----------------------
'''

from bottle import route, run, template, request
import urllib2

import numpy as np
import numpy
import cv2
import os
import pylab
import imageio

import ast

import smtplib
import os,email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
#from email.Utils import COMMSPACE,formatdate
from email import Encoders



import matplotlib.pyplot as plt
import matplotlib
import argparse
import math
import pymorph
from PIL import ImageChops

import statistics

from datetime import datetime, time
import matlab.engine

import ast
import pickle

import itertools
import subprocess
import csv

import os
import glob

import re

import random
numbers = re.compile(r'(\d+)')



def send_mail(send_from, send_to, subject, text, filenames):
    
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Date'] = " Use any date time module to insert or use email.utils formatdate"
    msg['Subject'] = subject
    
    msg.attach( MIMEText(text) )
    for f in filenames:
        part = MIMEBase('application', "octet-stream")
        fo=open(f,"rb")
        part.set_payload(fo.read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(f))
        msg.attach(part)

    smtp = smtplib.SMTP("smtp.gmail.com",587) #Email id  for yahoo use smtp.mail.yahoo.com
    smtp.ehlo()
    smtp.starttls()  #in yahoo use smtplib.SMTP_SSL()
    smtp.ehlo()
    smtp.login("gmcpalgorithm@gmail.com","gmcprocks") #Edit
    sent=smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
    return sent

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))

def midpoint(x1, y1, x2, y2):
    x = (x1+x2)/2
    y = (y1+y2)/2
    return [x,y]
def three(x1, y1, x2, y2):
    x = (x1+x2+((x1+x2)/2))/3
    y = (y1+y2+((y1+y2)/2))/3
    point2 = midpoint(x,y,x2,y2)
    return [[x,y], point2]

def four(x1, y1, x2, y2):
    x = (x1+x2+((x1+x2)/2)+((x1+x2)/2))/4
    y = (y1+y2+((y1+y2)/2)+((y1+y2)/2))/4
    array = three(x,y,x2,y2)
    return [[x,y], array[0], array[1]]

def five(x1, y1, x2, y2):
    x = (x1+x2+((x1+x2)/2)+((x1+x2)/2)+((x1+x2)/2))/5
    y = (y1+y2+((y1+y2)/2)+((y1+y2)/2)+((y1+y2)/2))/5
    array = four(x,y,x2,y2)
    return [[x,y], array[0], array[1], array[2]]


def draw_detections(img, rects, index, pointarray, detectionarray, clusterpoints, histarray,thickness = 1):
    for x, y, w, h in rects:
        print "Center at:"+str((x+w/2,y+h/2))
        xpoint = x+w/2
        ypoint = y+h/2
        point = []
        point.append(xpoint)
        point.append(ypoint)
        clusterpoints.append(point)
        point = []
        if index==0:
            xpoint = xpoint
            ypoint = ypoint
        if index ==1:
            xpoint = xpoint+1920
        if index ==2:
            xpoint = xpoint+3840
        if index==3:
            ypoint = ypoint+1080
        if index==4:
            ypoint = ypoint+1080
            xpoint = xpoint+1920
        if index==5:
            ypoint = ypoint+1080
            xpoint = xpoint+3840
        pointarray = []
        pointarray.append(xpoint)
        pointarray.append(ypoint)
        detectionarray.append(pointarray)
        pointarray = []
        #cv2.circle(img, (x+w/2,y+h/2), 1, (0, 0, 255), thickness=5, lineType=8, shift=0)
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        x1 = x+pad_w
        y1 = y+pad_w
        x2 = x+w-pad_w
        y2 = y+h-pad_h
        crop = img[y1:y2,x1:x2]
        cv2.imshow("crop", crop)
        cv2.waitKey(0)
        image = crop
        hist = cv2.calcHist(image, [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()
        #print hist
        histarray.append(hist)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
    return detectionarray, clusterpoints, histarray

def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
 
    # return the chi-squared distance
    return d
def chunks(l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

@route('/home', method='POST')
def home():
    data = request.body.read()
    #print data
    data1 = data.split('&')
    print data1
    #print data
    data = data1[0]
    begin = data1[1]

    emp = False
    #print begin
    step = data1[2]
    ratio = data1[3]
    email = data1[4]
    startframe = data1[5]
    data = data.split('=')
    data = data[1]
    data = urllib2.unquote(data)
    data = data[:-1]
    data = data +"1"
    url = data
    begin = begin.split('=')
    begin = begin[1]
    if begin == "":
        begin = 0
        emp = True
    begin = int(begin)
    step = step.split('=')
    step = step[1]
    if step == "":
        step = 15
    step = int(step)
    ratio = ratio.split('=')
    ratio = ratio[1]
    ratio = float(ratio)
    email = email.split('=')
    email = email[1]
    email = urllib2.unquote(email)
    email = str(email)
    if email == "":
        return "No email"
    startframe = startframe.split('=')
    startframe = startframe[1]
    if startframe == "":
        startframe = 0
    startframe = int(startframe)
    print ratio
    print email
    print url
    print "Start: "+str(startframe)+" End: "+str(begin)
    u = urllib2.urlopen(url)
    data1 = u.read()
    u.close()
    with open("video.mp4", "wb") as f :
        f.write(data1)
    print "Recieved this link: " + str(data)

    #lot of changes required

    cap = cv2.VideoCapture("video.mp4")
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    Frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)


    print int(Frames)

    if emp:
        begin = int(Frames)
    blank_image = Image.new("RGB", (5760, 2160))
        
    filename = 'video.mp4'
    vid = imageio.get_reader(filename,  'ffmpeg')

    start = 0
    trackstep = step
    vcounter = 0

    smallstep = trackstep/5
    smallstep = int(smallstep)

    files = glob.glob('trackvideos/*')
    for f in files:
        os.remove(f)

    filename = "trackletdetections.csv"
    # opening the file with w+ mode truncates the file
    f = open(filename, "w+")
    f.close()
    for x in xrange(startframe,begin,trackstep):
        vcounter = vcounter+1
        begval = x
        step = smallstep
        bigstep =trackstep
        hypdist = 80
        trajectories = []
        alltracks = []
        nums = []
        num = []
        nums = [begval,begval+step,begval+step*2,begval+step*3,begval+step*4,begval+step*5]
        pointarray = []
        detectionarray = []
        framearray = []
        clusterarray = []
        clusterpoints = []
        indexcounter = 0
        histarray = []
        histogramarray=[]

        drframes = []

        donehyp = False
        points_hist = []
        points_cluster=[]
        points_frame=[]
        points_box = []

        print nums
        boundbox = []
        boundingpoints = []
        boxesarray = []

        path = os.path.dirname(os.path.realpath(__file__))
        eng = matlab.engine.start_matlab()
        eng.cd('voc-release4.01')
        print "Starting"
        for num in nums:
            image = vid.get_data(num)
            index1 = nums.index(num)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if indexcounter ==0:
                firstframe = image
            if indexcounter ==5:
                lastframe = image
            cv2.imwrite('saveim.jpeg',image)
            im = image.copy()
            image = "/saveim.jpeg"
            himage = cv2.imread('saveim.jpeg')
            pathim = str(path)+str(image)
            val = eng.persontest({'arg1': pathim})
            counter = 0
            newlist = []
            personlist = []
            for s in val:
                for c in s:
                    counter = counter+1
                    if counter<37:
                        c = round(c,2)
                        newlist.append(c)
                personlist.append(newlist)
                newlist = []
                counter = 0

            newperson = []
            newpersonlist = []
            for s in personlist:
                newperson = list(chunks(s,4))
                newpersonlist.append(newperson)

            #image = cv2.imread('frame.jpg')
            bcounter = 0
            histarray = []
            bighist = []
            pointy=[]
            arrayofpoints = []
            for person in newpersonlist:
                #print "===================================="
                #print "Person Coordinates: "+str(person[0])
                personcoord = person[0]
                x1 = personcoord[0]
                y1 = personcoord[1]
                x2 = personcoord[2]
                y2 = personcoord[3]
                xpoint = (x1+x2)/2
                ypoint = (y1+y2)/2
                pointy.append(xpoint)
                pointy.append(ypoint)
                boundbox = [[x1,y1],[x2,y2]]
                boundingpoints.append(boundbox)
                clusterpoints.append(pointy)
                pointy=[]
                if index1==0:
                    xpoint = xpoint
                    ypoint = ypoint
                if index1 ==1:
                    xpoint = xpoint+1920
                if index1 ==2:
                    xpoint = xpoint+3840
                if index1==3:
                    ypoint = ypoint+1080
                if index1==4:
                    ypoint = ypoint+1080
                    xpoint = xpoint+1920
                if index1==5:
                    ypoint = ypoint+1080
                    xpoint = xpoint+3840
                pointy.append(xpoint)
                pointy.append(ypoint)
                arrayofpoints.append(pointy)
                pointy = []
                #print "yo"
                for body in person:
                    bcounter = bcounter+1
                    if bcounter>1:
                        #print "Body Coordinate"+" "+str(bcounter)+": "+str(body)
                        x1 = body[0]
                        y1 = body[1]
                        x2 = body[2]
                        y2 = body[3]
                        cropbody = himage[y1:y2,x1:x2]
                        cropbody = cv2.cvtColor(cropbody, cv2.COLOR_BGR2RGB)
                        try:
                            hist4 = cv2.calcHist(cropbody, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
                            hist4 = cv2.normalize(hist4).flatten()
                        except:
                            #cv2.imshow("crop",cropbody)
                            #cv2.waitKey(0)
                            hist4 = []
                            hist4 = numpy.array(hist4)
                        histarray.append(hist4)
                bighist.append(histarray)
                histarray = []
                bcounter = 0
                '''
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

            img = image
            found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)

            index1 = nums.index(num)
            arrayofpoints, clusterpoints, histarray = draw_detections(img, found, index1, pointarray, detectionarray, clusterpoints,histarray, 2)
            '''
            if indexcounter ==0:
                seedarray = clusterpoints
            #print arrayofpoints
            framearray.append(arrayofpoints)
            boxesarray.append(boundingpoints)
            clusterarray.append(clusterpoints)
            histogramarray.append(bighist)
            detectionarray = []
            arrayofpoints = []
            clusterpoints = [] 
            boundingpoints = []
            histarray = []
            #print found

            #cv2.imshow('img', img)
            #cv2.waitKey(0)
            print "finished image " +str(indexcounter)
            img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            indexcounter = indexcounter +1
            
            pilimg= Image.fromarray(img)
            index = nums.index(num)
            blank_image.paste(pilimg, (1920*index,0))


            if index >=3:
                nloc = index-3
                blank_image.paste(pilimg, (1920*nloc,1080))

            blank_image.save("blank.jpg")
            
            
            
            
        #eng.quit()
        #print framearray

        for f in framearray:
            for p in f:
                points_frame.append(p)

        for c in clusterarray:
            for p in c:
                points_cluster.append(p)

        for b in boxesarray:
            for bp in b:
                points_box.append(bp)


        safeties = [[0,0],[450,0],[900,0],[0,800],[450,800],[900,800]]
        counterhist = 0
        for h in histogramarray:
            for c in h:
                counterhist = counterhist+1
                if counterhist==1:
                    safetyhist1 = c
                if counterhist==2:
                    safetyhist2 = c
                if counterhist==3:
                    safetyhist3 = c
                if counterhist==4:
                    safetyhist4 = c
                if counterhist==5:
                    safetyhist5 = c
                if counterhist==6:
                    safetyhist6 = c
                points_hist.append(c)
        counterhist =0
        '''
        #adding safety points
        framearray[0].append([0,0])
        points_frame.append([0,0])
        points_cluster.append([0,0])
        points_hist.append(safetyhist1)

        framearray[1].append([1920,0])
        points_frame.append([1920,0])
        points_cluster.append([0,1])
        points_hist.append(safetyhist2)

        framearray[2].append([3840,0])
        points_frame.append([3840,0])
        points_cluster.append([1,1])
        points_hist.append(safetyhist3)

        framearray[3].append([0,1080])
        points_frame.append([0,1080])
        points_cluster.append([1,2])
        points_hist.append(safetyhist4)

        framearray[4].append([1920,1080])
        points_frame.append([1920,1080])
        points_cluster.append([2,2])
        points_hist.append(safetyhist5)

        framearray[5].append([3840,1080])
        points_frame.append([3840,1080])
        points_cluster.append([2,3])
        points_hist.append(safetyhist6)
        '''
        #create input graph

        connections = 0
        dupcount = 0
        drawnline = []
        drawnarray = []
        pointarray = []

        oblank = numpy.array(blank_image)
        mclique = oblank.copy()
        blanky = oblank.copy()
        blanky = cv2.cvtColor(oblank, cv2.COLOR_RGB2BGR)
        mclique = cv2.cvtColor(mclique, cv2.COLOR_RGB2BGR)
        oblank = cv2.cvtColor(oblank, cv2.COLOR_RGB2BGR)
        oblank = numpy.array(oblank)
        cv2.line(oblank,(10,10),(100,100),(255,0,0),1)
        print framearray
        print "length of framearray begin"
        print len(framearray)
        framecopy1 = list(framearray)

        print framearray

        outfile = open('frame.pkl', 'wb')
        pickle.dump(framecopy1, outfile)
        outfile.close()
        tracklets1 =[]

        donecounter =0
        framees = []


        beginvectorframe = [framearray[0][0],framearray[1][0],framearray[2][0],framearray[3][0],framearray[4][0],framearray[5][0]]

        for point in beginvectorframe:
            index = points_frame.index(point)
            val = points_cluster[index]

        arrayofdistances1 = []
        distancep = 0
        evaluated = False
        hypotnodes = []
        counterarray = 0
        poslengths = []
        while True:

            try:
                arrayofdistances1 = []
                distancep = 0
            


                f0 = random.choice(framearray[0])
                index = points_frame.index(f0)
                c0 = points_cluster[index]
                f1 = random.choice(framearray[1])
                index = points_frame.index(f1)
                c1 = points_cluster[index]
                f2 = random.choice(framearray[2])
                index = points_frame.index(f2)
                c2 = points_cluster[index]
                f3 = random.choice(framearray[3])
                index = points_frame.index(f3)
                c3 = points_cluster[index]
                f4 = random.choice(framearray[4])
                index = points_frame.index(f4)
                c4 = points_cluster[index]
                f5 = random.choice(framearray[5])
                index = points_frame.index(f5)
                c5 = points_cluster[index]


                arr = [c0,c1,c2,c3,c4,c5]

                print "Inputted array "+str(arr)
                def minimumclique(arr,rcount):
                    nochange = False
                    distancep = 0
                    toedit = list(arr)
                    firstcost = 0

                    arrayofdistances1 = []
                    arrayofarrays = []


                    arrayofarrays.append(toedit)
                    array = toedit
                    for point in array:
                        curpoint = point
                        index = points_cluster.index(curpoint)
                        #caluclate real cluster point
                        hist = points_hist[index]
                        for p in array:
                            if p!=curpoint:
                                index4 = array.index(curpoint)
                                if index4 ==0:
                                    if p in hypotnodes:
                                        distancep = distancep + 1000
                                    pindex = array.index(p)
                                    npindex = pindex-1
                                    pp = array[npindex]
                                    index = points_cluster.index(p)
                                        #caluclate real cluster point
                                    hist2 = points_hist[index]

                                    #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                    
                                    try:
                                        d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400

                                        d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                        distancep = distancep + d*400
                                    except:
                                        continue
                                    

                                    distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                        
                    entiredistance = distancep
                    firstcost = entiredistance
                    #print "First Cost: "+ str(entiredistance)
                    arrayofdistances1.append(entiredistance)
                    entiredistance = 0
                    distancep = 0


                    toedit = list(arr)
                    #print "toedit: "+str(toedit)
                    for val in framearray[0]:
                        toedit = list(arr)
                        index = points_frame.index(val)
                        v = points_cluster[index]

                        toedit[0] = v
                        #print toedit

                        arrayofarrays.append(toedit)


                        #calculate clique
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            #caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p!=curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 ==0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex-1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                            #caluclate real cluster point
                                        hist2 = points_hist[index]

                                        #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                        
                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400
                                        except:
                                            continue
                                        

                                        distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                            
                        entiredistance = distancep
                        #print "Cost: "+str(entiredistance)
                        arrayofdistances1.append(entiredistance)
                        
                        entiredistance = 0
                        distancep = 0

                    toedit = list(arr)
                    #print "toedit: "+str(toedit)
                    for val in framearray[1]:
                        toedit = list(arr)
                        index = points_frame.index(val)
                        v = points_cluster[index]

                        toedit[1] = v
                        #print toedit
                        arrayofarrays.append(toedit)

                        #calculate clique
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            #caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p!=curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 ==0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex-1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                            #caluclate real cluster point
                                        hist2 = points_hist[index]

                                        #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                        
                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400
                                        except:
                                            continue
                                        

                                        distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                            
                        entiredistance = distancep
                        #print "Cost: "+str(entiredistance)
                        arrayofdistances1.append(entiredistance)
                        entiredistance = 0
                        distancep = 0

                    toedit = list(arr)
                    #print "toedit: "+str(toedit)
                    for val in framearray[2]:
                        toedit = list(arr)
                        index = points_frame.index(val)
                        v = points_cluster[index]
                        toedit[2] = v
                        #print toedit
                        arrayofarrays.append(toedit)

                        #calculate clique
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            #caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p!=curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 ==0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex-1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                            #caluclate real cluster point
                                        hist2 = points_hist[index]

                                        #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                        
                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400
                                        except:
                                            continue
                                        

                                        distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                            
                        entiredistance = distancep
                        #print "Cost: "+str(entiredistance)
                        arrayofdistances1.append(entiredistance)
                        entiredistance = 0
                        distancep = 0

                    toedit = list(arr)
                    #print "toedit: "+str(toedit)
                    for val in framearray[3]:
                        toedit = list(arr)
                        index = points_frame.index(val)
                        v = points_cluster[index]
                        toedit[3] = v
                        #print toedit
                        arrayofarrays.append(toedit)

                        #calculate clique
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            #caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p!=curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 ==0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex-1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                            #caluclate real cluster point
                                        hist2 = points_hist[index]

                                        #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                        
                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400
                                        except:
                                            continue
                                        

                                        distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                            
                        entiredistance = distancep
                        #print "Cost: "+str(entiredistance)
                        arrayofdistances1.append(entiredistance)
                        entiredistance = 0
                        distancep = 0

                    toedit = list(arr)
                    #print "toedit: "+str(toedit)
                    for val in framearray[4]:
                        toedit = list(arr)
                        index = points_frame.index(val)
                        v = points_cluster[index]

                        toedit[4] = v
                        #print toedit
                        arrayofarrays.append(toedit)

                        #calculate clique
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            #caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p!=curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 ==0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex-1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                            #caluclate real cluster point
                                        hist2 = points_hist[index]

                                        #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                        
                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400
                                        except:
                                            continue
                                        

                                        distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                            
                        entiredistance = distancep
                        #print "Cost: "+str(entiredistance)
                        arrayofdistances1.append(entiredistance)
                        entiredistance = 0
                        distancep = 0

                    toedit = list(arr)
                    #print "toedit: "+str(toedit)
                    for val in framearray[5]:
                        toedit = list(arr)
                        index = points_frame.index(val)
                        v = points_cluster[index]

                        toedit[5] = v
                        #print toedit

                        arrayofarrays.append(toedit)

                        #calculate clique
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            #caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p!=curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 ==0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex-1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                            #caluclate real cluster point
                                        hist2 = points_hist[index]

                                        #histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...
                                        
                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.cv.CV_COMP_BHATTACHARYYA)
                                            distancep = distancep + d*400
                                        except:
                                            continue
                                        

                                        distancep = distancep + distance(pp[0],pp[1],p[0],p[1])
                            
                        entiredistance = distancep
                        #print "Cost: "+str(entiredistance)
                        arrayofdistances1.append(entiredistance)
                        entiredistance = 0
                        distancep = 0
                    minimum = min(arrayofdistances1)

                    print "FIRST COST:"+str(firstcost)
                    print "MINIMUM:" +str(minimum)

                    index = arrayofdistances1.index(minimum)


                    minimum_path = arrayofarrays[index]

                    print "CHOSEN PATH: "+str(minimum_path)
                    

                    #print "Current array " + str(arr)

                    dif = firstcost-minimum
                    rcount = rcount+1
                    if dif<100 or rcount==5:
                        print "returning minimum_path "+ str(minimum_path)
                        return minimum_path
                    else:
                        print "calling again with minimum_path "+str(minimum_path)
                        return minimumclique(minimum_path, rcount)
                rcount = 0
                minimum_clique = minimumclique(arr, rcount)

                print "=============+"
                print minimum_clique
                print "=============+"

                minimum_path = minimum_clique


                print donehyp
                print minimum_path

                seed = minimum_path[0]
                point1 = minimum_path[1]
                point2 = minimum_path[2]
                point3 = minimum_path[3]
                point4 = minimum_path[4]
                point5 = minimum_path[5]

                pointsfortrack = []
                pointsfortrack.append(seed)
                pointsfortrack.append(point1)
                pointsfortrack.append(point2)
                pointsfortrack.append(point3)
                pointsfortrack.append(point4)
                pointsfortrack.append(point5)

                tracklets1.append(pointsfortrack)
                pointsfortrack = []

                index = points_cluster.index(seed)
                #caluclate real cluster point
                seed = points_frame[index]

                index = points_cluster.index(point1)
                #caluclate real cluster point
                point1 = points_frame[index]

                index = points_cluster.index(point2)
                #caluclate real cluster point
                point2 = points_frame[index]

                index = points_cluster.index(point3)
                #caluclate real cluster point
                point3 = points_frame[index]

                index = points_cluster.index(point4)
                #caluclate real cluster point
                point4 = points_frame[index]

                index = points_cluster.index(point5)
                #caluclate real cluster point
                point5 = points_frame[index]
             

                etracklet = list(minimum_path)
                print"=========================================="
                print minimum_path
                print"=========================================="
                clist = minimum_path
                #print clist[0][0]
                d1 = distance(clist[0][0],clist[0][1],clist[1][0],clist[1][1])

                d2 = distance(clist[1][0],clist[1][1],clist[2][0],clist[2][1])

                d3 = distance(clist[2][0],clist[2][1],clist[3][0],clist[3][1])

                d4 = distance(clist[3][0],clist[3][1],clist[4][0],clist[4][1])

                d5 = distance(clist[4][0],clist[4][1],clist[5][0],clist[5][1])

                cslopes = [d1,d2,d3,d4,d5]
                #print clist
                
                sureshots = []
                for s in cslopes:
                    if s>hypdist:
                        for i, val in enumerate(cslopes):
                            if s==val:
                                sureshots.append(i)
                print sureshots
                surehypots = []
                if sureshots != []:
                    if len(sureshots)==1:
                        s = sureshots[0]
                        surehypots = minimum_path[s+1]
                        surehypots = [surehypots]
                    else:
                        beg = sureshots[0]
                        beg = beg+1
                        last = sureshots[-1]
                        last = last+1
                        if beg==5:
                            surehypots = minimum_path[-1]
                            surehypots = [surehypots]
                        if beg==4 and last==5:
                            print "last 2"
                            surehypots = minimum_path[-2:]
                            surehypots = surehypots
                        else:
                            surehypots = minimum_path[beg:last]
                    print "Sureshot hypothetical node required here: "+str(surehypots)
                toremove = []
                pointarray=[]
                
                indexes = []
                print "length of framearray"
                print len(framearray)
                copy = list(framearray)
                for array in framearray:
                    for point in array:
                        if point==seed:
                            i1 = framearray.index(array)
                            i2 = array.index(point)
                            indexes.append(i1)
                            indexes.append(i2)
                            toremove.append(indexes)
                            indexes = []
                        if point==point1:
                            i1 = framearray.index(array)
                            i2 = array.index(point)
                            indexes.append(i1)
                            indexes.append(i2)
                            toremove.append(indexes)
                            indexes = []
                        if point==point2:
                            i1 = framearray.index(array)
                            i2 = array.index(point)
                            indexes.append(i1)
                            indexes.append(i2)
                            toremove.append(indexes)
                            indexes = []
                        if point==point3:
                            i1 = framearray.index(array)
                            i2 = array.index(point)
                            indexes.append(i1)
                            indexes.append(i2)
                            toremove.append(indexes)
                            indexes = []
                        if point==point4:
                            i1 = framearray.index(array)
                            i2 = array.index(point)
                            indexes.append(i1)
                            indexes.append(i2)
                            toremove.append(indexes)
                            indexes = []
                        if point==point5:
                            i1 = framearray.index(array)
                            i2 = array.index(point)
                            indexes.append(i1)
                            indexes.append(i2)
                            toremove.append(indexes)
                            indexes = []
                print toremove
                toremove2 = list(toremove)
                toremove.sort()
                allpos2 =  list(toremove2 for toremove2,_ in itertools.groupby(toremove2))
                toremove = allpos2
                for indexes in toremove:
                    i1 = indexes[0]
                    i2 = indexes[1]
                    try: 
                        point = framearray[i1][i2]
                        ind = points_frame.index(point)
                        cpoint = points_cluster[ind]
                        if cpoint in surehypots:
                            if cpoint in drframes:
                                try:
                                    print "removing "+str(framearray[i1][i2])
                                    del framearray[i1][i2]
                                except:
                                    pass
                            else:
                                print "!!!!!!!!!!!!!!!!!!!!"
                                print "found a saver "+ str(cpoint)
                                print "!!!!!!!!!!!!!!!!!!!!"
                                pass
                        else:
                            try:
                                print "removing "+str(framearray[i1][i2])
                                del framearray[i1][i2]
                            except:
                                print "unable to remove "+str(framearray[i1][i2])


                                
                    except:
                        print "houston we got problem"
                print len(framearray)
                print framearray

                for i, l in enumerate(poslengths):
                    if i>3:
                        l1 = poslengths[i]
                        l2 = poslengths[i-1]
                        l3 = poslengths[i-2]

                        if l1==l2 and l2==l3:
                            print "endless loop detected"
                            for indexes in toremove:
                                i1 = indexes[0]
                                i2 = indexes[1]
                                del framearray[i1][i2]
                            break


                print donehyp
                

                print "before"
                print donehyp
                empty = False
                for frame in framearray:
                    if frame==[]:
                        empty = True
                s =[]
                if empty and donehyp==False:
                    for frame in framearray:
                        if frame != []:
                            fl = len(frame)
                            s.append(fl)
                    avg = sum(s)/len(s)
                    avg = int(avg)
                    avg = avg +1
                    print "~~~~~~"
                    print avg
                    print "~~~~~~"
                    for i, frame in enumerate(framearray):
                        if frame ==[]:
                            for x in range(0,avg):
                                if i ==0:
                                    framearray[i].append([0,x])
                                    framees.append([i,[0,x]])
                                    drframes.append([0,x])
                                    points_frame.append([0,x])
                                    points_cluster.append([0,x])
                                    points_hist.append(safetyhist1)
                                if i ==1:
                                    framearray[i].append([2370,x])
                                    framees.append([i,[2370,x]])
                                    points_frame.append([2370,x])
                                    points_cluster.append([450,x])
                                    drframes.append([450,x])
                                    points_hist.append(safetyhist1)
                                if i==2:
                                    framees.append([i,[4740,x]])
                                    framearray[i].append([4740,x])
                                    points_frame.append([4740,x])
                                    points_cluster.append([900,x])
                                    drframes.append([900,x])
                                    points_hist.append(safetyhist3)
                                if i==3:
                                    framees.append([i,[0,1880+x]])
                                    framearray[i].append([0,1880+x])
                                    points_frame.append([0,1880+x])
                                    points_cluster.append([0,800+x])
                                    drframes.append([0,800+x])
                                    points_hist.append(safetyhist4)
                                if i==4:
                                    framees.append([i,[2370,1880+x]])
                                    framearray[i].append([2370,1880+x])
                                    points_frame.append([2370,1880+x])
                                    points_cluster.append([450,800+x])
                                    drframes.append([450,800+x])
                                    points_hist.append(safetyhist5)
                                if i==5:
                                    framees.append([i,[4740,1880+x]])
                                    framearray[i].append([4740,1880+x])
                                    points_frame.append([4740,1880+x])
                                    points_cluster.append([900,800+x])
                                    drframes.append([900,800+x])
                                    points_hist.append(safetyhist6)
                        

                    donehyp = True
                    
            except Exception,e: 
                print str(e)
                break

            

        oblank = cv2.cvtColor(oblank, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Asdf",oblank)
        #cv2.waitKey(0)
        lined = Image.fromarray(oblank)
        lined.save("lines.jpg")
        indices = []
        newtracklet = []
        tracklets2 = []
        pi = 0

        f = open('frame.pkl', 'rb')
        framecopy2 = pickle.load(f)

        framecopy2[0].append([0,0])
        framecopy2[1].append([1920,0])
        framecopy2[2].append([3840,0])
        framecopy2[3].append([0,1080])
        framecopy2[4].append([1920,1080])
        framecopy2[5].append([3840,1080])


        for f in framees:
            if f[0]==0:
                framecopy2[0].append(f[1])
            if f[0]==1:
                framecopy2[1].append(f[1])
            if f[0]==2:
                framecopy2[2].append(f[1])
            if f[0]==3:
                framecopy2[3].append(f[1])
            if f[0]==4:
                framecopy2[4].append(f[1])
            if f[0]==5:
                framecopy2[5].append(f[1])

        
            
        firstframe = Image.fromarray(firstframe)
        lastframe = Image.fromarray(lastframe)
        blended = Image.blend(lastframe,firstframe,0)
        blended = numpy.array(blended)

        safteycounter = 0
        poplist = []
        for track in tracklets1:
            for point in track:
                if point in safeties:
                    safteycounter = safteycounter+1
            index = tracklets1.index(track)
            if safteycounter>3:
                poplist.append(index)
            safteycounter = 0

        for i in poplist:
            tracklets1.pop(i)

        list1 = [[0, 0], [450, 0], [900, 0], [0, 800], [450, 800], [900, 800]]

        if list1 in tracklets1:
            tracklets1.remove(list1)

        colors = [(212,0,162),(58,81,232),(0,212,14),(0,240,255),(232,81,58),(0,168,255),(255,255,255),(212,0,162),(58,81,232),(0,212,14),(0,198,212),(198,0,212),(0,168,255),(255,255,255),(209,209,0),(0,255,179),(212,0,162),(58,81,232),(0,212,14),(0,240,255),(232,81,58),(0,168,255),(255,255,255)]
        curtrack = []
        alltracks = []
        for tracklet in tracklets1:
            
            alltracks.append(tracklet)

        blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        title = "tracklets.jpg"

        trackletsonblended = Image.fromarray(blended)
        trackletsonblended.save(title)

            
        trajs = alltracks

        for t in trajs:
            print "++++++++++++++++++++++++"
            print t
            print "++++++++++++++++++++++++"


        ntrajs = []
        def hypothetical(trajectory, cin):
            ntrajs = []
            trajs = trajectory
            for tracklet in trajs:
                etracklet = list(tracklet)
                print"=========================================="
                print tracklet
                print"=========================================="
                clist = tracklet
                #print clist[0][0]
                d1 = distance(clist[0][0],clist[0][1],clist[1][0],clist[1][1])

                d2 = distance(clist[1][0],clist[1][1],clist[2][0],clist[2][1])

                d3 = distance(clist[2][0],clist[2][1],clist[3][0],clist[3][1])

                d4 = distance(clist[3][0],clist[3][1],clist[4][0],clist[4][1])

                d5 = distance(clist[4][0],clist[4][1],clist[5][0],clist[5][1])

                cslopes = [d1,d2,d3,d4,d5]
                #print clist
                
                print cslopes
                '''
                asc = sorted(cslopes, key=int) 
                dif = numpy.percentile(asc, 75) - numpy.percentile(asc, 25)
                qrange = dif *0.5
                #print qrange
                med = statistics.median(cslopes)
                inds = []
                for v in cslopes:
                    dif = abs(v-med)
                    if dif>qrange and v>med:
                        c=0
                        for i, val in enumerate(cslopes):
                            if c>0:
                                break
                            if val ==v:
                                if i in inds:
                                    pass
                                else:
                                    #print inds
                                    ind = i
                                    #print "appending "+ str(ind)
                                    inds.append(ind)
                                    c = c+1
                        outlier = tracklet[ind]
                        #print "Outlier distance at index:"+str(ind)
                '''
                sureshots = []
                for s in cslopes:
                    if s>hypdist:
                        for i, val in enumerate(cslopes):
                            if s==val:
                                sureshots.append(i)
                '''
                for i in inds:
                    if cslopes[i] >hypdist:
                        print i
                        #sureshot messup
                        sureshots.append(i)
                '''
                print sureshots
                surehypots = []
                if sureshots != []:
                    if len(sureshots)==1:
                        s = sureshots[0]
                        surehypots = tracklet[s+1]
                        surehypots = [surehypots]
                    else:
                        beg = sureshots[0]
                        beg = beg+1
                        last = sureshots[-1]
                        last = last+1
                        if beg==5:
                            surehypots = tracklet[-1]
                            surehypots = [surehypots]
                        if beg==4 and last==5:
                            print "last 2"
                            surehypots = tracklet[-2:]
                            surehypots = surehypots
                        else:
                            surehypots = tracklet[beg:last]
                    print "Sureshot hypothetical node required here: "+str(surehypots)
                for hyp in surehypots:
                    etracklet.remove(hyp)
                print "Inliers: "+str(etracklet)
                print "Outliers: "+str(surehypots)
                for hyp in surehypots:
                    index = tracklet.index(hyp)
                    if len(etracklet)<3:
                        val1 = etracklet[0]
                        val2 = etracklet[1]
                        changex = val2[0]-val1[0]
                        changey = val2[1]-val1[1]
                    else:
                        try:
                            if index-2<0 or index-1<0:
                                print ""
                                raise ValueError('A very specific bad thing happened')
                            val1 = etracklet[index-2]
                            val2 = etracklet[index-1]
                            changex = val2[0]-val1[0]
                            changey = val2[1]-val1[1]
                        except:
                            val1 = etracklet[index]
                            val2 = etracklet[index+1]
                            changex = val2[0]-val1[0]
                            changey = val2[1]-val1[1]
                    try:
                        samp = etracklet[index-1]
                        hypotx = samp[0]+changex
                        hypoty = samp[1]+changey
                        hypot = [hypotx,hypoty]
                    except:
                        samp = val1
                        hypotx = samp[0]-changex
                        hypoty = samp[1]-changey
                        hypot = [hypotx,hypoty]
                    print "inserting "+str(hypot)
                    etracklet.insert(index, hypot)
                print etracklet
                ntrajs.append(etracklet)
            if cin == 6:
                return ntrajs
            if cin < 6:
                cin = cin+1
                return hypothetical(ntrajs,cin)

        print "**********"
        print "DISTANCES"
        print "**********"

        ntrajs = hypothetical(trajs,0)
        print ntrajs
        for tracklet in ntrajs:
            print "+++++++++++++++++++++++"
            print tracklet
            print "+++++++++++++++++++++++"

        mtrajs = []
        for tracklet in ntrajs:
            seed= tracklet[0]
            point1= tracklet[1]
            point2= tracklet[2]
            point3= tracklet[3]
            point4= tracklet[4]
            point5= tracklet[5]
            xaverage = (seed[0]+point1[0]+point2[0]+point3[0]+point4[0]+point5[0])/6
            yaverage = (seed[1]+point1[1]+point2[1]+point3[1]+point4[1]+point5[1])/6

            average = [xaverage,yaverage]
            mtrajs.append([seed,average,point5])

        buffer1 = hypdist+20
        nope = False
        toremove = []
        for tracklet in ntrajs:
            nope = False
            clist = tracklet
            #print clist[0][0]
            d1 = distance(clist[0][0],clist[0][1],clist[1][0],clist[1][1])

            d2 = distance(clist[1][0],clist[1][1],clist[2][0],clist[2][1])

            d3 = distance(clist[2][0],clist[2][1],clist[3][0],clist[3][1])

            d4 = distance(clist[3][0],clist[3][1],clist[4][0],clist[4][1])

            d5 = distance(clist[4][0],clist[4][1],clist[5][0],clist[5][1])

            cslopes = [d1,d2,d3,d4,d5]
            for s in cslopes:
                if s>buffer1:
                    nope = True
            if nope:
                toremove.append(tracklet)

        for t in toremove:
            ntrajs.remove(t)

        toremove2 = []
        for tracklet in ntrajs:
            p1 = tracklet
            for t in ntrajs:
                if t!=p1:
                    p2 = t
                    #compare p1 and p2
                    if p1[0]==p2[0]:
                        #battle
                        p1counter = 0
                        p2counter = 0
                        p1points = []
                        p2points = []
                        for point in p1:
                            if point in points_cluster:
                                if point in p1points:
                                    pass
                                else:
                                    p1points.append(point)
                                    p1counter = p1counter+1
                        for point in p2:
                            if point in points_cluster:
                                if point in p2points:
                                    pass
                                else:
                                    p2points.append(point)
                                    p2counter = p2counter+1
                        if p1counter>p2counter:
                            #remove p2
                            toremove2.append(p2)
                        else:
                            pass 

        for tr in toremove2:
            try:
                ntrajs.remove(tr)
            except:
                pass


        #removing duplicates
        try:
            ntrajs.sort()
            allpos =  list(ntrajs for ntrajs,_ in itertools.groupby(ntrajs))
            ntrajs = allpos
        except Exception,e: 
            print str(e)




        toremove3 = []
        for t in ntrajs:
            pcounter = 0
            for point in t:
                if point in points_cluster:
                    if point in drframes:
                        pass
                    else:
                        pcounter = pcounter +1
            if pcounter>0:
                pass
            else:
                toremove3.append(t)

        for t in toremove3:
            ntrajs.remove(t)

        toremove4 = []

        for t in ntrajs:
            counterp = 0
            for point in t:
                if point in points_cluster:
                    counterp = counterp+1
            if counterp<=1:
                toremove4.append(t)
            if counterp==2:
                p1 = t[0]
                for tp in ntrajs:
                    p2 = tp[0]
                    dist = distance(p1[0],p1[1],p2[0],p2[1])
                    if dist<10:
                        toremove4.append(t)


        for t in toremove4:
            try:
                ntrajs.remove(t)
            except:
                pass


        for t in ntrajs:
            print "*******************"
            print t
            print "*******************"

        averages = []
        for tracklet in ntrajs:
            xdists = []
            ydists = []
            for point in tracklet:
                if point in points_cluster:

                    index = points_cluster.index(point)
                    box = points_box[index]

                    p1 = box[0]
                    p2 = box[1]

                    x1 = p1[0]
                    y1 = p1[1]

                    x2 = p2[0]
                    y2 = p2[1]

                    xm = point[0]
                    ym = point[1]

                    xdist = x2 - xm
                    ydist = y2 - ym

                    xdists.append(xdist)
                    ydists.append(ydist)

            xaverage = sum(xdists)/len(xdists)
            yaverage = sum(ydists)/len(ydists)

            avgvector = []
            avgvector.append(xaverage)
            avgvector.append(yaverage)

            averages.append(avgvector)

        '''
        Frame #    ID    Detection   Box W,L 
        '''

        c = csv.writer(open("trackletdetections.csv", "ab"))
        videostring = "video" + str(vcounter) + ".mp4"
        c.writerow([videostring, " "])
        c.writerow(["Frame #","ID (tracklet-specific)","Detection (Center)","Bounding Box (Width,Height)"])
        for frame in nums:
            framenum = frame
            ind = nums.index(frame)
            for t in ntrajs:
                det = t[ind]
                detid = ntrajs.index(t)
                distances = averages[detid]
                width = distances[0]+distances[0]
                length = distances[1]+distances[1]
                dim = "("+str(width)+","+str(length)+")"
                c.writerow([framenum,detid,det,dim])


        path = '/Users/Utkarsh/Documents/GMCP_Website/'
        video = "video.mp4"
        cap = cv2.VideoCapture(path+video)


        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1920)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        w=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
        h=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))

        print w,h
        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        out = cv2.VideoWriter('tudout.mp4',fourcc, 15, (w,h))
        counter = 0

        #editable
        start = begval

        second = start+step
        third = start +step*2
        fourth = start+step*3
        fifth = start+step*4
        sixth = start+step*5

        middle = start+step*2.5
        last = start+step*5
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        counter5 = 0

        #print start
        #print middle
        #print last
        while True:
            ret, frame = cap.read()
            counter = counter +1
            if ret==True:

                #frame = cv2.resize(frame, (1920, 1080)) 
                for trajectories in ntrajs:
                    index = ntrajs.index(trajectories)

                    distances = averages[index]

                    xdist = distances[0]
                    ydist = distances[1]

                    color = colors[index]
                    if counter==start:
                        cv2.rectangle(frame, (int(trajectories[0][0]-xdist), int(trajectories[0][1]-ydist)), (int(trajectories[0][0]+xdist), int(trajectories[0][1]+ydist)), color, 4)
                    #print fcounter
                    if start<counter<second:
                        counter1 = counter-start
                        #print fcounterss
                        dist = second-start
                        curprog = counter1/float(dist)
                        x1 = trajectories[0][0]
                        x2 = trajectories[1][0]
                        difx = x2-x1
                        curx = float(difx*curprog)
                        curx = x1+curx
                        y1 = trajectories[0][1]
                        y2 = trajectories[1][1]
                        dify = y2-y1
                        cury = float(dify*curprog)
                        cury = y1+cury
                        cv2.rectangle(frame, (int(curx-xdist), int(cury-ydist)), (int(curx+xdist), int(cury+ydist)), color, 4)
                        #cv2.imshow("frame",frame)
                        #cv2.waitKey(0)
                    if counter==second:
                        cv2.rectangle(frame, (int(trajectories[1][0]-xdist), int(trajectories[1][1]-ydist)), (int(trajectories[1][0]+xdist), int(trajectories[1][1]+ydist)), color, 4)
                    if second<counter<third:
                        counter2 = counter-second
                        dist = third-second
                        curprog = counter2/float(dist)
                        x1 = trajectories[1][0]
                        x2 = trajectories[2][0]
                        difx = x2-x1
                        curx = float(difx*curprog)
                        curx = x1+curx
                        y1 = trajectories[1][1]
                        y2 = trajectories[2][1]
                        dify = y2-y1
                        cury = float(dify*curprog)
                        cury = y1+cury
                        cv2.rectangle(frame, (int(curx-xdist), int(cury-ydist)), (int(curx+xdist), int(cury+ydist)), color, 4)
                    if counter==third:
                        cv2.rectangle(frame, (int(trajectories[2][0]-xdist), int(trajectories[2][1]-ydist)), (int(trajectories[2][0]+xdist), int(trajectories[2][1]+ydist)), color, 4)
                    if third<counter<fourth:
                        counter3 = counter-third
                        dist = fourth-third
                        curprog = counter3/float(dist)
                        x1 = trajectories[2][0]
                        x2 = trajectories[3][0]
                        difx = x2-x1
                        curx = float(difx*curprog)
                        curx = x1+curx
                        y1 = trajectories[2][1]
                        y2 = trajectories[3][1]
                        dify = y2-y1
                        cury = float(dify*curprog)
                        cury = y1+cury
                        cv2.rectangle(frame, (int(curx-xdist), int(cury-ydist)), (int(curx+xdist), int(cury+ydist)), color, 4)
                    if counter==fourth:
                        cv2.rectangle(frame, (int(trajectories[3][0]-xdist), int(trajectories[3][1]-ydist)), (int(trajectories[3][0]+xdist), int(trajectories[3][1]+ydist)), color, 4)
                    if fourth<counter<fifth:
                        counter4 = counter-fourth
                        dist = fifth-fourth
                        curprog = counter4/float(dist)
                        x1 = trajectories[3][0]
                        x2 = trajectories[4][0]
                        difx = x2-x1
                        curx = float(difx*curprog)
                        curx = x1+curx
                        y1 = trajectories[3][1]
                        y2 = trajectories[4][1]
                        dify = y2-y1
                        cury = float(dify*curprog)
                        cury = y1+cury
                        cv2.rectangle(frame, (int(curx-xdist), int(cury-ydist)), (int(curx+xdist), int(cury+ydist)), color, 4)
                    if counter==fifth:
                        cv2.rectangle(frame, (int(trajectories[4][0]-xdist), int(trajectories[4][1]-ydist)), (int(trajectories[4][0]+xdist), int(trajectories[4][1]+ydist)), color, 4)
                    if fifth<counter<sixth:
                        counter5 = counter-fifth
                        dist = sixth-fifth
                        curprog = counter5/float(dist)
                        x1 = trajectories[4][0]
                        x2 = trajectories[5][0]
                        difx = x2-x1
                        curx = float(difx*curprog)
                        curx = x1+curx
                        y1 = trajectories[4][1]
                        y2 = trajectories[5][1]
                        dify = y2-y1
                        cury = float(dify*curprog)
                        cury = y1+cury
                        cv2.rectangle(frame, (int(curx-xdist), int(cury-ydist)), (int(curx+xdist), int(cury+ydist)), color, 4)
                    if counter==sixth:
                        cv2.rectangle(frame, (int(trajectories[5][0]-xdist), int(trajectories[5][1]-ydist)), (int(trajectories[5][0]+xdist), int(trajectories[5][1]+ydist)), color, 4)
                    
                #fcounter = 0
                #scounter = 0

                # write the flipped frame

                #frame = cv2.resize(frame, (1440, 1080)) 
                if counter>=start:
                    out.write(frame)

                if counter>=last:
                    break
                '''
                cv2.imshow('frame',frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                '''
            else:
                pass


        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        vidstring = "ffmpeg -i tudout.mp4 trackvideos/video"+str(vcounter)+".mp4"
        subprocess.call(vidstring,shell=True) 

        '''

        path = '/Users/Utkarsh/Documents/GMCP_Python/'
        video = 'video.mp4'
        cap = cv2.VideoCapture(path+video)


        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1920)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        w=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
        h=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))

        print w,h
        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        out = cv2.VideoWriter('tudout.mp4',fourcc, 15, (w,h))
        counter = 0

        #editable
        start = begval

        second = start+step
        third = start +step*2
        fourth = start+step*3
        fifth = start+step*4
        sixth = start+step*5

        middle = start+step*2.5
        last = start+step*5
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        counter5 = 0

        #print start
        #print middle
        #print last
        while True:
            ret, frame = cap.read()
            counter = counter +1
            if ret==True:

                #frame = cv2.resize(frame, (1920, 1080)) 
                for trajectories in ntrajs:
                    index = ntrajs.index(trajectories)

                    distances = averages[index]

                    xdist = distances[0]
                    ydist = distances[1]

                    color = (0,0,255)
                    if counter==start:
                        if trajectories[0] in points_cluster:
                            cv2.rectangle(frame, (int(trajectories[0][0]-xdist), int(trajectories[0][1]-ydist)), (int(trajectories[0][0]+xdist), int(trajectories[0][1]+ydist)), color, 4)

                    if counter==second:
                        if trajectories[1] in points_cluster:
                            cv2.rectangle(frame, (int(trajectories[1][0]-xdist), int(trajectories[1][1]-ydist)), (int(trajectories[1][0]+xdist), int(trajectories[1][1]+ydist)), color, 4)

                    if counter==third:
                        if trajectories[2] in points_cluster:
                            cv2.rectangle(frame, (int(trajectories[2][0]-xdist), int(trajectories[2][1]-ydist)), (int(trajectories[2][0]+xdist), int(trajectories[2][1]+ydist)), color, 4)

                    if counter==fourth:
                        if trajectories[3] in points_cluster:
                            cv2.rectangle(frame, (int(trajectories[3][0]-xdist), int(trajectories[3][1]-ydist)), (int(trajectories[3][0]+xdist), int(trajectories[3][1]+ydist)), color, 4)

                    if counter==fifth:
                        if trajectories[4] in points_cluster:
                            cv2.rectangle(frame, (int(trajectories[4][0]-xdist), int(trajectories[4][1]-ydist)), (int(trajectories[4][0]+xdist), int(trajectories[4][1]+ydist)), color, 4)

                    if counter==sixth:
                        if trajectories[5] in points_cluster:
                            cv2.rectangle(frame, (int(trajectories[5][0]-xdist), int(trajectories[5][1]-ydist)), (int(trajectories[5][0]+xdist), int(trajectories[5][1]+ydist)), color, 4)
                    
                #fcounter = 0
                #scounter = 0

                # write the flipped frame

                #frame = cv2.resize(frame, (1440, 1080)) 
                if counter>=start:
                    out.write(frame)

                if counter>=last:
                    break
                
                cv2.imshow('frame',frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            else:
                pass


        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        vidstring = "ffmpeg -i tudout.mp4 trackvideos/video"+str(vcounter+1)+".mp4"
        subprocess.call(vidstring,shell=True) 
        '''


    files = glob.glob('trackvideos/*')

    files2 = []
    for f in files:
        f = f[12:]
        files2.append(f)




    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    sorted_files = sorted(files2, key=numericalSort)

    with open("test2.txt", "w") as text_file:
        for f in sorted_files:
            text_file.write(f+str("\n"))

    filenames = []
    def sorted_ls(path):
        mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
        return list(sorted(os.listdir(path), key=mtime))

    listing2 = sorted_ls('trackvideos/')

    for infile in listing2:
        filenames.append("trackvideos/"+infile)

    filenames.append("trackletdetections.csv")


    s=send_mail("gmcpalgorithm@gmail.com",email,"Your Tracked Video","Attached are your tracklets in video form. Each human is surrounded with different colors.",filenames)  # Edit
    if (s.keys()==[]):
        print "Message Sent!"
        return "Tracked Video Sent to: " +str(email)
    else:
        print "Error!"
        return "Error When Sending Email"
    

run(host='localhost', port=8080, debug=True)
run(reloader=True)