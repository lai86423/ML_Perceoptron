#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tkinter import *
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import random


# In[2]:


class Perceptron():
    def __init__(self,dataset,epoch,learning_rate):
        self.X=dataset
        self.bias=-1
        self.random_w = np.array([random.random(),random.random(),self.bias])#w初始值(0,1)
        self.P_w = self.random_w
        self.learning_rate = learning_rate
        self.N = epoch
        self.train_X=[]
        self.test_X=[]
        self.train_Y=[]
        self.test_Y=[]
        self.train_d=[]#期望輸出
        self.test_d=[]#期望輸出
        self.train_m=0
        self.test_m=0
        self.train_Accuracy=0.0
        self.test_Accuracy=0.0
        self.TrainNum=0
        self.Adapted_train_Y=[]

    def set_data(self):
        #打亂資料
        self.X=np.random.permutation(self.X) #print("打亂後資料\n",X) #<class 'numpy.ndarray'>
        self.X=np.array(self.X) 
        #計算輸入檔案之數量 維度 row,col 
        m,n=np.shape(self.X) 
        n=n-1#扣掉最後一筆是期望輸出
        print("所有資料數和維度",m,n)

        # 檢查是否二類問題
        if(n>2):
            print("非二類問題")
            tk.messagebox.showinfo("非二類問題","非二類問題")
            return 

        #訓練資料和期望輸出的切割
        temp_X=np.array_split(self.X,n,axis=1)#將最後一筆期望輸出切出
        temp_d=temp_X[1]
        temp_X=temp_X[0]

        x0=-(np.ones(m))#X運算時需減掉閥值 用X0=-1來運算
        #將x0加在資料最後一筆
        temp_X=np.column_stack((temp_X,x0))#記得 加在最後一筆 跟課本是加在第0筆
        #print(temp_X)

        #切割訓練與測試資料
        self.train_m=round((m/3)*2) #訓練資料數2/3
        self.test_m=m-self.train_m #測試資料1/3 #print(train_m,test_m)
        self.train_X=temp_X[:self.train_m]
        self.test_X=temp_X[self.train_m:]
        #print("訓練資料=",train_X,"測試資料",test_X)

        #切割訓練與測試預期輸出
        self.train_d=temp_d[:self.train_m]
        self.test_d=temp_d[self.train_m:]
        train_temp = []
        test_temp = []
        for i in self.train_d:
            for j in i:
                train_temp.append(j)

        for x in self.test_d:
            for u in x:
                test_temp.append(u)

        self.train_d=np.array(train_temp)
        self.test_d=np.array(test_temp)
        print("訓練預期輸出=",self.train_d,"測試預期輸出=",self.test_d)
        self.train_Y=np.zeros(int(self.train_m)) #實際輸出 預設0 #print(train_Y)
        self.test_Y=np.zeros(int(self.test_m))
        #print("train_Y=",train_Y,"test_Y=",test_Y)         

        # label非0/1組合 改變label-> 0~1
        if (0 not in self.train_d) or (1 not in self.train_d):
            for i in range(int(self.train_m)):
                self.train_d[i]=self.train_d[i]%2
        if (0 not in self.test_d) or (1 not in self.test_d):
            for i in range(int(self.test_m)):
                self.test_d[i]=self.test_d[i]%2     
        print("***修改0/1後****訓練預期輸出=",self.train_d,"測試預期輸出=",self.test_d)
        
    def sgn(self,y):
        if y > 0:
            return 1
        else:
            return 0    
    #訓練資料
    def Percetron_Learning(self):
        #P_w=np.array([0,1,-1])#w初始值(0,1)  閥值視為最後一筆 (課本的w0)
        self.TrainNum=0
        AllCorrect=False
        print("閥值,收斂條件,學習率=",self.P_w,self.N,self.learning_rate)          
        for n in range(self.N):   
            if(AllCorrect==False):
                for i in range(int(self.train_m)):
                    print("第%d回的第%d次訓練，值為"%(n+1,i+1),self.train_X[i,:])            
                    print("w與x取內積值=",self.P_w.dot(self.train_X[i,:]))
                    self.train_Y[i]=self.sgn(self.P_w.dot(self.train_X[i,:])) # y=sign((w．X))
                    print("經活化函數後w．x 的值",self.train_Y[i])
                    print("y[i]=",self.train_Y[i],"d[i]=",self.train_d[i])#測
                    print("W=",self.P_w)
                    if(self.train_Y[i]!=self.train_d[i]):
                        if(self.train_Y[i]<self.train_d[i]):
                            self.P_w=self.P_w+self.learning_rate*self.train_X[i,:] #+或-學習率判斷 ，由乘上期望輸出的正負號即可知
                        else:
                            self.P_w=self.P_w-self.learning_rate*self.train_X[i,:] #+或-學習率判斷 ，由乘上期望輸出的正負號即可知                    
                            #w_record.append(P_w.copy())
                            self.TrainNum+=1
                            print("W第"+str(self.TrainNum)+"次修正=",self.P_w)
                            continue                      
                        if np.all(self.train_Y==self.train_d):
                            print("提前修正!")
                            AllCorrect=True
                            break
            print("w最終為",self.P_w)
            
    def Accuracy(self,A_x,A_y,A_d,m,final_w):    
        print("***計算辨識率***")
        Error=0
        for i in range(int(m)):
            print("第%d筆資料="%(i+1),A_x[i,:])            
            print("w與x取內積值=",final_w.dot(A_x[i,:]))
            A_y[i]=self.sgn((final_w.dot(A_x[i,:]))) # y=sign((w．X))
            print("經活化函數後w．x 的值",A_y[i])
            #print("y[i]=",A_y[i],"d[i]=",A_d[i])#測
            if(A_y[i]!=A_d[i]):
                Error+=1
        Accuracy=((m-Error))*100/m
        print("Error=",Error,"M=",m,"Accuracy==",Accuracy)
        print(Accuracy)
        return Accuracy  
            


# In[3]:


#datafile_list= ['perceptron1.txt','perceptron2.txt','2Ccircle1.txt','2Circle1.txt',
#                        '2Circle2.txt','2CloseS.txt','2CloseS2.txt','2CloseS3.txt','2cring.txt',
#                        '2CS.txt','2Hcircle1.txt','2ring.txt','Number.txt']
#filename = 'dataSet\\2ring.txt'
#with open(filename,'r') as f :
#    #讀資料 
#    for line in f :
#        X.append(list(map(float, line.split(' '))))       


# In[4]:


def get_data2(self,X):
    #  filename = askopenfilename()
    #  with open(filename,'r') as f :
        #讀資料 
        #  for line in f :
          #    X.append(list(map(float, line.split(' '))))   
    #打亂資料
    X=np.random.permutation(X) #print("打亂後資料\n",X) #<class 'numpy.ndarray'>
    X=np.array(X) 
    #計算輸入檔案之數量 維度 row,col 
    m,n=np.shape(X) 
    n=n-1#扣掉最後一筆是期望輸出
    print("所有資料數和維度",m,n)

    # 檢查是否二類問題
    if(n>2):
        print("非二類問題")
        tk.messagebox.showinfo("非二類問題","非二類問題")
        return 

    #訓練資料和期望輸出的切割
    temp_X=np.array_split(X,n,axis=1)#將最後一筆期望輸出切出
    temp_d=temp_X[1]
    temp_X=temp_X[0]

    x0=-(np.ones(m))#X運算時需減掉閥值 用X0=-1來運算
    #將x0加在資料最後一筆
    temp_X=np.column_stack((temp_X,x0))#記得 加在最後一筆 跟課本是加在第0筆
    #print(temp_X)

    #切割訓練與測試資料
    self.train_m=round((m/3)*2) #訓練資料數2/3
    self.test_m=m-self.train_m #測試資料1/3 #print(train_m,test_m)
    self.train_X=temp_X[:self.train_m]
    self.test_X=temp_X[self.train_m:]
    #print("訓練資料=",train_X,"測試資料",test_X)

    #切割訓練與測試預期輸出
    self.train_d=temp_d[:self.train_m]
    self.test_d=temp_d[self.train_m:]
    train_temp = []
    test_temp = []
    for i in self.train_d:
        for j in i:
            train_temp.append(j)

    for x in self.test_d:
        for u in x:
            test_temp.append(u)

    self.train_d=np.array(train_temp)
    self.test_d=np.array(test_temp)
    print("訓練預期輸出=",self.train_d,"測試預期輸出=",self.test_d)
    self.train_Y=np.zeros(int(self.train_m)) #實際輸出 預設0 #print(train_Y)
    self.test_Y=np.zeros(int(self.test_m))
    #print("train_Y=",train_Y,"test_Y=",test_Y)         

    # label非0/1組合 改變label-> 0~1
    if (0 not in self.train_d) or (1 not in self.train_d):
        for i in range(int(self.train_m)):
            self.train_d[i]=self.train_d[i]%2
    if (0 not in self.test_d) or (1 not in self.test_d):
        for i in range(int(self.test_m)):
            self.test_d[i]=self.test_d[i]%2     
    print("修改0/1後訓練預期輸出=",self.train_d,"測試預期輸出=",self.test_d)


# In[5]:


def sgn2(y):
    if y > 0:
        return 1
    else:
        return 0    


# In[6]:


#訓練資料
def Percetron_Learning2(x,y,m,d,learning_rate):
        #P_w=np.array([0,1,-1])#w初始值(0,1)  閥值視為最後一筆 (課本的w0)
        WchangeNum=0
        AllCorrect=False
        print("閥值,收斂條件,學習率=",P_w,N,learning_rate)          
        for n in range(N):   
            if(AllCorrect==False):
                for i in range(int(m)):
                    print("第%d回的第%d次訓練，值為"%(n+1,i+1),x[i,:])            
                    print("w與x取內積值=",P_w.dot(x[i,:]))
                    y[i]=sgn(P_w.dot(x[i,:])) # y=sign((w．X))
                    print("經活化函數後w．x 的值",y[i])
                    print("y[i]=",y[i],"d[i]=",d[i])#測
                    print("W=",P_w)
                    if(y[i]!=d[i]):
                        if(y[i]<d[i]):
                            P_w=P_w+learning_rate*x[i,:] #+或-學習率判斷 ，由乘上期望輸出的正負號即可知
                        else:
                            P_w=P_w-learning_rate*x[i,:] #+或-學習率判斷 ，由乘上期望輸出的正負號即可知                    
                        w_record.append(P_w.copy())
                        WchangeNum+=1
                        print("W第"+str(WchangeNum)+"次修正=",P_w)
                        continue                      
                    if np.all(y==d):
                        print("提前修正!")
                        AllCorrect=True
                        break
        print("w最終為",P_w)
        Adapted_Y=y
        return P_w,Adapted_Y,WchangeNum


# In[7]:


def Accuracy2(A_x,A_y,A_d,m,final_w):    
    Error=0
    for i in range(int(m)):
        print("第%d筆資料="%(i+1),A_x[i,:])            
        print("w與x取內積值=",final_w.dot(A_x[i,:]))
        A_y[i]=sgn(final_w.dot(A_x[i,:])) # y=sign((w．X))
        print("經活化函數後w．x 的值",A_y[i])
        print("y[i]=",A_y[i],"d[i]=",A_d[i])#測
        if(A_y[i]!=A_d[i]):
            Error+=1

    A=((m-Error))*100/m
    print("Error=",Error,"M=",m,"Accuracy==",A)
    print(A)
    return A     

