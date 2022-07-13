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
from nnHw1_perceptron import Perceptron


# In[2]:


class Application(tk.Frame):
    
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.window = master
        self.grid()
        self.drawGUI()
    
    def drawGUI(self):
        # 設定學習率
        self.learning_rate_label = tk.Label(self)
        self.learning_rate_label["text"] = "輸入學習率"
        self.learning_rate_label.grid(row=0, column=0, sticky=tk.N+tk.W)# sticky=tk.N+tk.W 保持水平居中
        
        #學習率輸入欄位
        self.learning_rate = tk.DoubleVar()
        self.learning_rate_entry = tk.Entry(self, textvariable=self.learning_rate)
        self.learning_rate_entry.grid(row=0, column=1, sticky=tk.N+tk.W)
        
        # 設定收斂條件
        self.epoch_label = tk.Label(self)
        self.epoch_label["text"] = "輸入收斂條件"
        self.epoch_label.grid(row=1, column=0, sticky=tk.N+tk.W)

        #收斂條件輸入欄位
        self.epoch = tk.IntVar()
        self.epoch_entry = tk.Entry(self, textvariable=self.epoch)
        self.epoch_entry.grid(row=1,column=1, sticky=tk.N+tk.W)
        
        # 選取檔案做訓練
        self.label = tk.Label(self)
        self.label["text"] = "載入資料集進行訓練測試"
        self.label.grid(row=3, column=0, sticky=tk.N+tk.W)
        
        # 選取檔案做訓練 按鈕
        self.load_data_button = tk.Button(self)
        self.load_data_button["text"] = "選取檔案"
        self.load_data_button.grid(row=3, column=1, sticky=tk.N+tk.W)
        self.load_data_button["command"] = self.get_data
        
        # 設定訓練圖
        
        self.training_data_figure = Figure(figsize=(3,3), dpi=100)
        #把绘制的图形显示到tkinter窗口上
        self.training_data_canvas = FigureCanvasTkAgg(self.training_data_figure, self)
        self.training_data_canvas.draw()
        self.training_data_canvas.get_tk_widget().grid(row=4, column=1, columnspan=3)
        
        #學習率=learning_rate訓練正確率=train_Accuracy測試正確率test_Accuracy
        # 結果文字輸出
        self.training_num_label = tk.Label(self)
        self.training_num_label["text"] = "實際訓練次數(Epoch)"
        self.training_num_label.grid(row=5, column=0, sticky=tk.N+tk.W)

        self.training_num_text_label = tk.Label(self)
        self.training_num_text_label["text"] = ""
        self.training_num_text_label.grid(row=5, column=1, sticky=tk.N+tk.W)

        self.training_acc_label = tk.Label(self)
        self.training_acc_label["text"] = "訓練辨識率(%)"
        self.training_acc_label.grid(row=6, column=0, sticky=tk.N+tk.W)

        self.training_acc_text_label = tk.Label(self)
        self.training_acc_text_label["text"] = ""
        self.training_acc_text_label.grid(row=6, column=1, sticky=tk.N+tk.W)

        self.testing_acc_label = tk.Label(self)
        self.testing_acc_label["text"] = "測試辨識率(%)"
        self.testing_acc_label.grid(row=7, column=0, sticky=tk.N+tk.W)

        self.testing_acc_text_label = tk.Label(self)
        self.testing_acc_text_label["text"] = ""
        self.testing_acc_text_label.grid(row=7, column=1, sticky=tk.N+tk.W)

        self.r_w_label = tk.Label(self)
        self.r_w_label["text"] = "初始隨機鍵結值 w1 w2 bias"
        self.r_w_label.grid(row=8, column=0, sticky=tk.N+tk.W)

        self.r_w_label_text_label = tk.Label(self)
        self.r_w_label_text_label["text"] = ""
        self.r_w_label_text_label.grid(row=8, column=1, sticky=tk.N+tk.W)
        
        self.w_label = tk.Label(self)
        self.w_label["text"] = "鍵結值 w1 w2 bias"
        self.w_label.grid(row=9, column=0, sticky=tk.N+tk.W)

        self.w_label_text_label = tk.Label(self)
        self.w_label_text_label["text"] = ""
        self.w_label_text_label.grid(row=9, column=1, sticky=tk.N+tk.W)
        
    def Draw_training_figure(self, training_dataset, testing_dataset,Adapted_train_Y, final_w,train_m,test_m): #training_dataset=train_X
        # 清空畫面
        self.training_data_figure.clf()
        self.training_data_figure.a = self.training_data_figure.add_subplot(111)#表示“1×1网格
        
       # 產生訓練資料並分成兩類
        X_0=[]
        Y_0=[]
        X_1=[]
        Y_1=[]
        for i in range (int(train_m)):
            if Adapted_train_Y[i]==0:
                X_0.append(training_dataset[i][0])
                Y_0.append(training_dataset[i][1])
            else:
                X_1.append(training_dataset[i][0])
                Y_1.append(training_dataset[i][1])
        # draw 全部資料集兩種分類資料的點位
        self.training_data_figure.a.plot(X_0, Y_0, 'co')
        self.training_data_figure.a.plot(X_1, Y_1, 'bo')

        # 產生測試資料
        X_test=[]
        Y_test=[]
        for i in range (int(test_m)):
                X_test.append(testing_dataset[i][0])
                Y_test.append(testing_dataset[i][1])
        
        # draw測試資料        
        self.training_data_figure.a.plot(X_test, Y_test, 'y+')
        
        # 保存全部資料集的畫布範圍
        xmin = self.training_data_figure.a.get_xlim()[0]
        xmax = self.training_data_figure.a.get_xlim()[1]
        ymin = self.training_data_figure.a.get_ylim()[0]
        ymax = self.training_data_figure.a.get_ylim()[1]   
        
        #畫切割線W
        x1 = np.arange(xmin-2,xmax+2,0.01)
        x2 = -(final_w[0]*x1-final_w[2])/final_w[1]
        line, = self.training_data_figure.a.plot(x1,x2, '-r', label='graph')     

        #畫布範圍
        self.training_data_figure.a.set_xlim(xmin-2,xmax+2,0.01)
        self.training_data_figure.a.set_ylim(ymin-2,ymax+2,0.01)

        self.training_data_figure.a.set_title('Traing Data')
        self.training_data_canvas.draw()

    def get_data(self):
        filename = askopenfilename()
        X=[]
        with open(filename,'r') as f :
            #讀資料 
            for line in f :
                X.append(list(map(float, line.split(' '))))   
            
        ##接收輸入學習率
        learning_rate = self.learning_rate.get()
        epoch = self.epoch.get()
        
        ##開始訓練
        print("*****開始訓練*****")
        percep=Perceptron(X,epoch,learning_rate)
        percep.set_data()
        
        self.r_w_label_text_label["text"] =percep.P_w
        
        percep.Percetron_Learning()
        training_acc=percep.Accuracy(percep.train_X,percep.train_Y,percep.train_d,percep.train_m,percep.P_w)
        print("*****訓練結束  計算辨識率*****")
        print("train_Accuracy=",training_acc)
        
        self.training_num_text_label["text"] = percep.TrainNum
        self.training_acc_text_label["text"] = training_acc
        self.w_label_text_label["text"] =percep.P_w
        
        #self.weight_text.delete(1.0, END) 
        
        testing_acc=percep.Accuracy(percep.test_X,percep.test_Y,percep.test_d,percep.test_m,percep.P_w)
        print("test_Accuracy=",testing_acc)
        

        self.testing_acc_text_label["text"] = testing_acc
        
        self.Draw_training_figure(percep.train_X,percep.test_X,percep.train_Y,percep.P_w,percep.train_m,percep.test_m)
        
window = tk.Tk()
app = Application(window)
window.mainloop()


# In[ ]:





# In[ ]:




