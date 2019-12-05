from multiprocessing import Process, Queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from serial import Serial
import time
from os import listdir
from os.path import isfile, join
import Queue as Q
import msvcrt
#from scipy.signal import savgol_filter

'''
result =63141.2687254
63341.2504158
63191.1516201
63241.1014749
65199.674103
63191.1516201
63643.5853369
65520.0678741
64000.0
63191.1516201
64000.0
64000.0
63191.1516201
64000.0
63241.1133941
67510.5517831
64000.0
63191.1516201
63191.1516201
dps
'''
def func1(queue,queue2):
    
    ser = Serial("COM6")
    data_buff_size = 8000           
    data = np.zeros(data_buff_size) 
    n_bytes = 16000
    n=2000
    nn=n+1
    k=0
    kk=0
    kkk=0
    data11=np.zeros(n)
    data22=np.zeros(n)
    data33=np.zeros(n)
    data44=np.zeros(n)
    data55=np.zeros(n)
    data66=np.zeros(n)
    data77=np.zeros(n)
    data88=np.zeros(n)
    
    k1=0

    
    def save_file(dmas,t2):

        files = [f for f in listdir('data_r3/take_1') if isfile(join('data_r3/take_1', f))]
        hh=len(files)+1
        hh1='data_r3/take_1/data_'+str(hh)+'.npy'
        y22 = dmas.astype(dtype=np.float16)
        np.save(hh1, y22)
        print "logging data for : ", time.time()-t2, " second"
        print "saving data as : ", hh1 

    
    def kb():
        x=msvcrt.kbhit()
        if x:
            xx=msvcrt.getch()
        else:
            return False
        return xx
    
    number=0        
    while k == 0:
        number+=1
        
        if number==1:
            time2=time.time()        
        
        if number ==21:
            number=0
            print (8000.0*20)/(time.time()-time2)
        
        
        kkk=kkk+1         
        rslt = ser.read(n_bytes)
        data = np.fromstring(rslt, dtype=np.uint16)
        ## print "ok ", time.time()-t1
        if data[0] > 9000:
            rslt=rslt[1:-1]
            data = np.fromstring(rslt, dtype=np.uint16)
            kk=ser.read(1)
        data1pp=data[0::8]
        data2pp=data[1::8]
        data3pp=data[2::8]
        data4pp=data[3::8]
        data5pp=data[4::8]
        data6pp=data[5::8]
        data7pp=data[6::8]
        data8pp=data[7::8]
            
        if data8pp[0] > 4400:
            data8p=data8pp-4500
            data7p=data7pp
            data6p=data6pp
            data5p=data5pp
            data4p=data4pp
            data3p=data3pp
            data2p=data2pp
            data1p=data1pp
        
        elif data7pp[0] > 4400:
            data8p=data7pp-4500
            data7p=data6pp
            data6p=data5pp
            data5p=data4pp
            data4p=data3pp
            data3p=data2pp
            data2p=data1pp
            data1p=data8pp
          
        elif data6pp[0] > 4400:
            data8p=data6pp-4500
            data7p=data5pp
            data6p=data4pp
            data5p=data3pp
            data4p=data2pp
            data3p=data1pp
            data2p=data8pp
            data1p=data7pp
        elif data5pp[0] > 4400:
            data8p=data5pp-4500
            data7p=data4pp
            data6p=data3pp
            data5p=data2pp
            data4p=data1pp
            data3p=data8pp
            data2p=data7pp
            data1p=data6pp
        elif data4pp[0] > 4400:
            data8p=data4pp-4500
            data7p=data3pp
            data6p=data2pp
            data5p=data1pp
            data4p=data8pp
            data3p=data7pp
            data2p=data6pp
            data1p=data5pp
        elif data3pp[0] > 4400:
            data8p=data3pp-4500
            data7p=data2pp
            data6p=data1pp
            data5p=data8pp
            data4p=data7pp
            data3p=data6pp
            data2p=data5pp
            data1p=data4pp
        elif data2pp[0] > 4400:
            data8p=data2pp-4500
            data7p=data1pp
            data6p=data8pp
            data5p=data7pp
            data4p=data6pp
            data3p=data5pp
            data2p=data4pp
            data1p=data3pp
        elif data1pp[0] > 4400:
            data8p=data1pp-4500
            data7p=data8pp
            data6p=data7pp
            data5p=data6pp
            data4p=data5pp
            data3p=data4pp
            data2p=data3pp
            data1p=data2pp
            

#        data1=data1p[0::20]
#        data2=data2p[0::20]
#        data3=data3p[0::20] 
#        data4=data4p[0::20] 
#        data5=data5p[0::20]

        data1=data1p
        data2=data2p
        data3=data3p 
        data4=data4p
        data5=data5p
        data6=data6p
        data7=data7p
        data8=data8p
                
        
        data11=np.append(data11,data1)
        data22=np.append(data22,data2)
        data33=np.append(data33,data3)
        data44=np.append(data44,data4)
        data55=np.append(data55,data5)
        data66=np.append(data66,data6)
        data77=np.append(data77,data7)
        data88=np.append(data88,data8)
        data11=data11[-n:]
        data22=data22[-n:]  
        data33=data33[-n:]
        data44=data44[-n:]
        data55=data55[-n:]
        data66=data66[-n:]
        data77=data77[-n:]
        data88=data88[-n:]
#        data333=data33[:-1]-data33[1:]
#        data333=data33[:-1]-data33[1:]
#        print data333.shape 
#        try :
#            n11=51
#            data33 = (savgol_filter(data33,n11, 3))
#        except IndexError:
#            data33=data33
        
        mydata=np.append([data11],[data22],axis=0)
        mydata=np.append(mydata,[data33],axis=0)
        mydata=np.append(mydata,[data44],axis=0)
        mydata=np.append(mydata,[data55],axis=0)
        mydata=np.append(mydata,[data66],axis=0)
        mydata=np.append(mydata,[data77],axis=0)
        mydata=np.append(mydata,[data88],axis=0)
        
        
        mydata1=mydata[:,0::3]
        queue.put(mydata1)

        try :
            x=queue2.get(timeout=0.001)
        except Q.Empty:
            x=4
        
        if x==0:
            print "begin data logging"
            k1=1
            dmas=np.append(mydata,mydata,1)
            t2=time.time()
        if x==1:
            k1=0
            save_file(dmas,t2)
            
        if k1==1:
            try:
                dmas=np.append(dmas,mydata,1)
            except ValueError:
                print data1.shape
                print data2.shape


def func2(queue):
#    from scipy.signal import savgol_filter
    
    time.sleep(0.4)
    win = pg.GraphicsWindow()
    win.setWindowTitle('Data, last 3 second')
    n=2000
    data11=np.zeros(n)
    data22=np.zeros(n)
    data33=np.zeros(n)
    data44=np.zeros(n)
    data55=np.zeros(n)
    data66=np.zeros(n)
    data77=np.zeros(n)
    data88=np.zeros(n)
    
    p1 = win.addPlot()#(colspan=2)
#    win.nextRow()
    p2 = win.addPlot()#(colspan=2)
    win.nextRow()
    p3 = win.addPlot()#(colspan=2)
#    win.nextRow()
    p4 = win.addPlot()#(colspan=2)
    win.nextRow()
    p5 = win.addPlot()#(colspan=2)
#    win.nextRow()
    p6 = win.addPlot()#(colspan=2)
    win.nextRow()
    p7 = win.addPlot()#(colspan=2)
#    win.nextRow()
    p8 = win.addPlot()#(colspan=2)        
    
    curve1 = p1.plot(data11, pen=(255,0,0), name="Original")
    curve2 = p2.plot(data22)
    curve3 = p3.plot(data33)
    curve4 = p4.plot(data44)
    curve5 = p5.plot(data55)
    curve6 = p6.plot(data66)
    curve7 = p7.plot(data77)
    curve8 = p8.plot(data88)
#    curve6 = p1.plot(data11, pen=(0,0,255), name="Filtered")
    

    
    def update1():

        mydata=queue.get(timeout=1)
        data11=mydata[0]
        data22=mydata[1]
        data33=mydata[2]
        data44=mydata[3]
        data55=mydata[4]
        data66=mydata[5]
        data77=mydata[6]
        data88=mydata[7]


        curve1.setData(data11)
        curve2.setData(data22)
        curve3.setData(data33)
        curve4.setData(data44)
        curve5.setData(data55)
        curve6.setData(data66)
        curve7.setData(data77)        
        curve8.setData(data88)


    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update1)
    timer.start(50)
    
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

#def func3(queue4):
#    import cv2
#    cap = cv2.VideoCapture(1)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    fgbg = cv2.createBackgroundSubtractorMOG2()
#    while True:
#        ret, frame = cap.read()
##        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        gray=frame
#        fgmask = fgbg.apply(gray)
#        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
##        
#        cv2.imshow('webcam original', frame)
#        cv2.imshow('webcam masked', fgmask)
##        cv2.imshow('gray', gray)
#        if cv2.waitKey(27) & 0xFF == ord('q'):
#            break

def func3(queue4):
    import cv2
    cap = cv2.VideoCapture(1)
    cap2=cv2.VideoCapture(2)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    fgbg = cv2.createBackgroundSubtractorMOG2()
    resize=np.zeros([120,160,3],dtype=int)
    resize2=np.zeros([120,160,3],dtype=int)
    resize3=np.zeros([120,160,3],dtype=int)
    resize4=np.zeros([120,160,3],dtype=int)   
#    frame2=np.zeros([600,640,3],dtype=int)
    
    while True:
        ret, frame = cap.read()
        retb, frameb = cap2.read()
        frame2 = cv2.resize(frame, (640, 600), interpolation = cv2.INTER_LINEAR)
        frameb2 = cv2.resize(frameb, (640, 600), interpolation = cv2.INTER_LINEAR)

        try :
            val1 = queue4.get(timeout=0.001)
            val11=val1[0]
        except Exception as error :
            val1=0
            val11=0

        if val11 > 0:
            print "get data"
            filecount1 = [f for f in listdir('data_r4/take1/pic') if isfile(join('data_r4/take1/pic', f))]
            nextfile=len(filecount1)+1
            pic='data_r4/take1/pic/pic'+str(nextfile)+'.jpg'
            picb='data_r4/take1/picb/picb'+str(nextfile)+'.jpg'
            cv2.imwrite(pic, frame)  
            cv2.imwrite(picb, frameb)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            speed=str(val1[0])
            speed=speed[:5]+' m/s'
            alt=str(val1[1]/10.0)
            alt=alt[:5]+' Percent'
            
            
            cv2.putText(frame,speed,(10,100), font, 1.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(frame,alt,(10,150), font, 1.5,(255,255,255),2,cv2.LINE_AA)
            resize4=resize3
            resize3=resize2
            resize2=resize
            resize = cv2.resize(frame, (160, 120), interpolation = cv2.INTER_LINEAR)
        frame2[0:120,0:160]=resize
        frame2[0:120,160:320]=resize2
        frame2[0:120,320:480]=resize3
        frame2[0:120,480:640]=resize4  
        frame2[120:600,0:640]=frame
        

#        font = cv2.FONT_HERSHEY_SIMPLEX
#        cv2.putText(frame,'OpenCV',(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        gray=frame
#        fgmask = fgbg.apply(gray)
#        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#        
        cv2.imshow('Cam original 1', frame2)
        cv2.imshow('Cam original 2', frameb2)
#        cv2.imshow('webcam masked', fgmask)
#        cv2.imshow('gray', gray)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break




def func4(queue2):
    
    import csv
    def kb():
        x=msvcrt.kbhit()
        if x:
            xx=msvcrt.getch()
        else:
            return False
        return xx
    k=0
    k2=1
    k3=0
    cons0=0
    cons1=1
    cons2=2
    cons3=3
    saveornot=0
    while k==0:
        x=kb()
        if x != False and x =='\r':
            print '\r'
            print x1
            print time.time()-ttt
#            queue5.put(x1)
        if x != False and x!='\r'and x !=';' and x !='<':
            if saveornot == 1:
                x=x.decode()
                print x,
                x1=x1+x
        if x != False and x =='<':
            queue2.put(cons1)
            saveornot=0
#            print '\r'
#            print x1
#            queue5.put(x1)
#            x1=x1+' took '+str(time.time()-t22)+' second'
            com=([x1, str(time.time()-t22)])
            files = [f for f in listdir('data_r3/take_1') if isfile(join('data_r3/take_1', f))]
            hh=len(files)+1
            hh='data_r3/take_1/comment/comment_'+str(hh)+'.csv'
            with open(hh,'w') as csvfile:
                w = csv.writer(csvfile)
                w.writerow(com)
        if x != False and x ==';':
            t22=time.time()
            x1="Description : "
            ttt=time.time()
            queue2.put(cons0)
            saveornot=1
        time.sleep(0.03)
        

if __name__ == '__main__':
    queue=Queue()
    queue2=Queue()
#    queue3=Queue()
    queue4=Queue()    

#    queue5=Queue()
#    queue6=Queue()
    
    p1 = Process(target=func1,args=(queue,queue2,))
    p1.start()
    p2 = Process(target=func2,args=(queue,))
    p2.start()
#    p3 = Process(target=func3,args=(queue4,))
#    p3.start()
    p4 = Process(target=func4,args=(queue2,))
    p4.start()
    
    p1.join()
    p2.join()
#    p3.join()
    p4.join()