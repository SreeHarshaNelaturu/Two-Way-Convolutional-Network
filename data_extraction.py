import cv2
import os
import cv2
import numpy as np
import pickle

'''
path = 'F:\Leisure\TV Series\Mr.Robot\Season 2\Mr.Robot.S02E01.720p.WEBRip.AAC2.0.H.264-KNiTTiNG[ettv]'

files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'Mr' in i]
#print(files)
'''

path_ann = "" # path of annotations
path_vid = "" # path of videos
path_pickle = "" #path where pickles are stored

ann_data = os.listdir(path_ann) #list of files in path_ann

vid_files = [i for i in os.listdir(path_vid) if os.path.isfile(os.path.join(path_vid,i))] #list of videos in path_vid

p = ""
for data in ann_data:
    p = path_ann + data
    
    file = open(p, "r") #opens annotation text file
    vid_name = data.rpartition('_')[0] #extracts whatever is written in front of the underscore from the name of textfile

    #finds the path of corresponding video
    for i in vid_files:
        if vid_name in i:
            vid = i
            break
    path = path_vid + "/" + vid #use backslash if required
    #arr = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and vid in i]
    cap = cv2.VideoCapture(path)
    
    for line in file:
        words = line.split()
        t1 = int(words[0]) #first frame
        t2 = int(words[1]) #last frame
        m = t1
        n = m + 7
        while n <= t2:
            pic = []
            for i in range(m,n):
                #reads ith frame, I am not sure if this will work,
                #if it dosent then we will have to calculate it with 
                #video parameters which are diffrenr for every video :(
                cap.set(1,i) 
                ret, frm = cap.read() 
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (720, 720))
                pic.append(gray)
                A = np.array(pic)
                #saves in path_pickle
                #I havent classified pickles according to lables yet, will do once this works
                with open(path_pickle, "wb") as f:
                   pickle.dump(A, f)    
            m = m + 1
            n = m + 7
        cap.release()

