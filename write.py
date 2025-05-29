import os
import tkinter
from tkinter import *
from tkinter import ttk, colorchooser, messagebox
from PIL import ImageGrab,Image
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf

color = '#C0BD30'
file_name = "img/1.jpg"
def svm(img):
    SZ=20
    bin_n = 16 # Number of bins
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    def deskew(img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
        img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
        return img
    def hog(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist
    img1 = cv2.imread('img/digits.png',0)
    if img1 is None:
        raise Exception("we need the digits.png image from samples/data here !")
    cells = [np.hsplit(row,100) for row in np.vsplit(img1,50)]
    # First half is trainData, remaining is testData
    train_cells = [ i[:50] for i in cells ]
    test_cells = [ i[50:] for i in cells]
    deskewed = [list(map(deskew,row)) for row in train_cells]
    hogdata = [list(map(hog,row)) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1,64)
    responses = np.repeat(np.arange(10),250)[:,np.newaxis]
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('model/svm_data.dat')

    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(20,20))
    deskewed = deskew(gray)
    hogdata = hog(deskewed)
    testData = np.float32(hogdata).reshape(-1,bin_n*4)
    result = svm.predict(testData)[1]#返回的第二个数为预测结果
    print(chr(int(result)+48))
    str.set(chr(int(result) + 48))
    messagebox.showinfo('识别结果：', "识别方法是：SVM\n 结果为:  {0}\n ".format(chr(int(result) + 48)))
    deskewed = [list(map(deskew,row)) for row in test_cells]
    hogdata = [list(map(hog,row)) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1,bin_n*4)
    result = svm.predict(testData)[1]#返回的第二个数为预测结果
    mask = result==responses
    correct = np.count_nonzero(mask)
    print('预测准确率为：',correct*100.0/result.size)
def KNN(img):
    img = cv2.imread('img/new.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (20, 20))
    out = np.array(gray)
    test = out.reshape(-1, 400).astype(np.float32)
    # Now load the data
    with np.load('model/knn_data_train.npz') as data:
        print(data.files)
        train = data['train']
        train_labels = data['train_labels']
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    print(result)
    str.set(chr(int(result) + 48))
    messagebox.showinfo('识别结果：', "识别方法是：KNN\n 结果为:  {0}\n ".format(chr(int(result) + 48)))
def CNN(img):
    with open('model/model_structure.json', 'r') as file:
        model_json1 = file.read()
    model = tf.keras.models.model_from_json(model_json1)
    model.load_weights('model/model_weight.h5')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    (thresh, out) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    test = out.reshape(1, 28, 28, 1).astype(np.float32)
    predictions1 = model.predict(test)
    print('识别到图片结果为：', np.argmax(predictions1))
    a=np.argmax(predictions1)
    str.set(a)
    messagebox.showinfo('识别结果：', "识别方法是：CNN\n 结果为:  {0}\n ".format(a))
def shibie():
    global path,result
    saveimg(canvas)
    img = cv2.imread('img/new.jpg')
    if v.get()=='SVM':
        svm(img)
    if v.get()=='KNN':
        KNN(img)
    if v.get()=='CNN':
        CNN(img)

def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def draw(event,color,sc):
    global lastx, lasty
    penw=sc.get()
    canvas.create_line(lastx,lasty,event.x,event.y,width=penw,fill=color,capstyle=ROUND)
    xy(event)

def colors():
    global color
    color=colorchooser.askcolor()[1]

def clear():
    cq=messagebox.askquestion(title='提示',message='是否清空绘图')
    if cq=='yes':
        canvas.delete(ALL)
def saveimg(widg):
    widg.update()
    x=root.winfo_x()+widg.winfo_x()+133
    y=root.winfo_y()+widg.winfo_y()+95
    x1=x+widg.winfo_width()+90
    y1=y+widg.winfo_height()+80
    print(x,y,x1,y1)
    ImageGrab.grab().crop((x,y,x1,y1)).save('img/new.jpg')
    image = Image.open('img/new.jpg')
    image.thumbnail((23, 23))
    image.save("img/1.jpg")  # 将图片保存为IMG/sum/1.jpg
    messagebox.showinfo('提示','已保存')

def Print(event):
    a = v.get()
    if a == 'KNN':
        print('1')
    if a == 'SVM':
        print('2')
    if a=='CNN':
        print('3')



# def xuanze():




def showwindo():
    global root,canvas,str,v
    file_name = "img/1.jpg"
    if os.path.exists(file_name):
        os.remove(file_name)
    root = Tk()
    str = StringVar()
    root.geometry('580x420+450+200')
    # 设置颜色

    '''
    设置按钮
    '''
    btnc = ttk.Button(root, text='颜色', command=colors,width=8)
    btnc.place(x=420, y=100)
    label1 = ttk.Label(root, text='笔宽',font=("微软雅黑", 11))
    label1.place(x=475, y=15)
    # 创建Sclae    https://blog.csdn.net/qiukapi/article/details/104068879
    cal = Scale(root,
                from_=15,  # 设置最小值
                to=80,  # 设置最大值
                orient=HORIZONTAL  # 设置水平方向
                )
    cal.place(x=440, y=35)
    btnc1 = ttk.Button(root, text='清空', command=clear,width=8)
    btnc1.place(x=500, y=100)
    btnc2 = ttk.Button(root, text='保存', command=lambda: saveimg(canvas),width=8)
    btnc2.place(x=420, y=380)
    btnc3 = ttk.Button(root, text='识别', command=shibie,width=8)
    btnc3.place(x=500, y=175)
    btnc4 = ttk.Button(root, text="退出", command=root.destroy,width=8)
    btnc4.place(x=500, y=380)

    label2 = ttk.Label(root, text='识别结果为:')
    label2.place(x=460, y=250)
    label1 = ttk.Label(root, textvariable=str, font=("微软雅黑", 28), foreground='red')
    label1.place(x=480, y=275)

    lable3=tk.Label(root, text='选择识别方法：')
    lable3.place(x=415, y=145)

    # 创建选项的列表
    Lang = ['KNN',
            'SVM',
            'CNN']
    v = tkinter.StringVar()
    v.set(Lang[0])
    om = tk.OptionMenu(root, v, *Lang)
    om.pack()
    om.place(x=420, y=175)
    om.bind('<Expose>', Print)
    '''
    创建画布

    '''
    canvas = Canvas(root, bg='#440053')  # bg设置背景颜色
    canvas.place(x=10, y=10, width=400, height=400)
    canvas.bind("<Button-1>", xy)
    canvas.bind("<B1-Motion>", lambda event: draw(event,color, cal))

    root.mainloop()

if __name__ == '__main__':
    showwindo()




