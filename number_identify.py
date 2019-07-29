import glob
import os
import sys
import numpy as np
import PIL.ImageOps
import tensorflow as tf
from PIL import Image
from tensorflow import keras

#global variables

def cutImg(lx,ly,rx,ry,arr):
    l=np.zeros(dtype=np.float,shape=(rx-lx+1,ry-ly+1))
    for i in range(lx,rx+1):
        for j in range(ly,ry+1):
            l[i-lx,j-ly]=arr[i,j]
    return l

def bfs(x,y,arr,height,width,visited):
    MinX=MaxX=x
    MinY=MaxY=y
    cnt=0
    from queue import Queue
    que=Queue()
    visited.add((x,y))
    que.put((x,y))
    mx=[0,1,0,-1]
    my=[1,0,-1,0]
    while (not que.empty()):
        (x,y)=que.get()
        MinX=min(MinX,x)
        MaxX=max(MaxX,x)
        MinY=min(MinY,y)
        MaxY=max(MaxY,y)
        cnt+=1
        for i in range(4): 
            tx=x+mx[i]
            ty=y+my[i]
            if (tx>=0 and tx<height and ty>=0 and ty<width and arr[tx,ty]>141 and ((tx,ty) not in visited)):
                visited.add((tx,ty))
                que.put((tx,ty))
    return (MinX,MaxX,MinY,MaxY,cnt)


def normalize(nparr):
    img=Image.fromarray(nparr)
    img=img.resize((28, 28),Image.ANTIALIAS)
    img=img.convert('L')
    img=np.array(img)
    
    '''rnt=np.zeros(shape=(28,28),dtype='float64')
    for i in range(20):
        for j in range(20):
            rnt[i+4,j+4]=img[i,j]/255.0'''
    return img.reshape(28,28,1)/255.0

def predictFromImg(img):
    print("processing image...")
    img=img.convert('L')
    img=PIL.ImageOps.invert(img)
    imgarr=np.array(img)
    (height,width)=imgarr.shape
    imglist=[]
    visited=set()
    for i in range(height):
        for j in range(width):
            if (imgarr[i,j]>141 and (i,j) not in visited):
                (MinX,MaxX,MinY,MaxY,cnt)=bfs(i,j,imgarr,height,width,visited)
                tot=(MaxX-MinX+1)*(MaxY-MinY+1)
                if tot>=784 and cnt>=500:
                    imglist.append((MinX,MinY,cutImg(MinX,MinY,MaxX,MaxY,imgarr)))

    def getKey(el):
        return (el[1],el[0])
    imglist.sort(key=getKey)
    predict_arr=[]
    for i in imglist:
        predict_arr.append(normalize(i[2]))
    predict_arr=np.array(predict_arr)
    print("loading model...")
    model=keras.models.load_model('mnist_model.h5')
    print("predicting...")
    results=model.predict(predict_arr)
    print("predict complete.")
    return results

def main():
    print("loading image...")
    try:
        filename=sys.argv[1]
        print("filename:",filename)
        img=Image.open(filename)
    except:
        print("Error occur when loading image, please check filepath.")
        return -1

    results=predictFromImg(img)

    print("The result is:")
    output=[]
    for i in results:
        if (np.max(i)>0.5):
            output.append(str(np.argmax(i)))
    print(','.join(output))

if __name__=='__main__':
    main()
