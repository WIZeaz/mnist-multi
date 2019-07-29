import os
import sys
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image,ImageTk
import number_identify as iden
import threading

class app(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title('number identify GUI')
        self.geometry('400x400')
        self.resizable(0,0)
        self.uFrame=Frame(self)
        self.uFrame.pack(side=TOP)

        self.lFrame=Frame(self.uFrame)
        self.lFrame.pack(side=LEFT)
        self.lstr=StringVar()
        self.lstr.set("identify from image: ")
        self.lLabel = Label(self.lFrame, textvariable=self.lstr, justify=CENTER)
        self.lLabel.pack(side=TOP)

        self.imgLabel = Label(self.lFrame)
        self.imgLabel.pack()

        self.lselectbutton=Button(self.lFrame,text="select image",command=self.selectImg)
        self.lselectbutton.pack()
        self.lidenbutton=Button(self.lFrame,text="identify",command=self.identify_thread)
        self.lidenbutton.pack()

        ''' self.rFrame=Frame(self.uFrame)
        self.rFrame.pack(side=LEFT)
        self.rstr=StringVar()
        self.rstr.set("identify from canvas: ")
        self.rLabel=Label(self.rFrame, textvariable=self.rstr, justify=CENTER)
        self.rLabel.pack(side=TOP)
        self.ridenbutton=Button(self.rFrame,text="identify")
        self.ridenbutton.pack()'''

        self.dFrame=Frame(self)
        self.dFrame.pack(side=BOTTOM)
        self.dstr=StringVar()
        self.dstr.set("")
        self.dLabel=Label(self.dFrame, textvariable=self.dstr, justify=CENTER)
        self.dLabel.pack(side=BOTTOM)

    def selectImg(self):
        file_path = filedialog.askopenfilename(title='select image')
        print(file_path)
        try:
            self.img=Image.open(file_path)
            width,height=self.img.size[0],self.img.size[1]
            if (width>300):
                scale=width/300.0
                width=int(width/scale+0.5)
                height=int(height/scale+0.5)
                self.photo=ImageTk.PhotoImage(self.img.resize((width, height),Image.ANTIALIAS))
            else:
                self.photo=ImageTk.PhotoImage(self.img)
            self.imgLabel.configure(image=self.photo)
        except:
           messagebox.showerror("Error","Something wrong happened when importing image.")
           return
    
    def identify_thread(self):
        self.dstr.set("predicting...please wait")
        t=threading.Thread(target=self.identify)
        t.start()

    def identify(self):
        results=iden.predictFromImg(self.img)
        output=[]
        for i in results:
            if (np.max(i)>0.5):
                output.append(str(np.argmax(i)))
        self.dstr.set("The results are: "+','.join(output))
    
        
if __name__=='__main__':
    Window=app()
    Window.mainloop()
