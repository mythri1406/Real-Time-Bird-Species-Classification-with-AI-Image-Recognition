
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import datetime
import cv2
import numpy as np

from tkinter.filedialog import askopenfilename

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
import skimage.io as io
import time
import cv2
import os
import matplotlib.patches as patches
import shutil



main = tkinter.Tk()
main.title("Bird Species Identification using Deep Learning")
main.geometry("1200x1200")

global filename

species_name = []
species_score = []

LR = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)

pretrain_model = 'inception_v3_iNat_299'
dataset = 'cub_200'

load_dir = os.path.join('./feature', pretrain_model)
features_train = np.load(os.path.join(load_dir, dataset + '_feature_train.npy'))
labels_train = np.load(os.path.join(load_dir, dataset + '_label_train.npy'))
features_val = np.load(os.path.join(load_dir, dataset + '_feature_val.npy'))
labels_val = np.load(os.path.join(load_dir, dataset + '_label_val.npy'))
print(features_train.shape)
print(labels_train.shape)
print(features_val.shape)
print(labels_val.shape)

tic = time.time()
LR.fit(features_train, labels_train)
labels_pred = LR.predict(features_val)

num_class = len(np.unique(labels_train))
acc = np.zeros((num_class, num_class), dtype=np.float32)
for i in range(len(labels_val)):
    acc[int(labels_val[i]), int(labels_pred[i])] += 1.0

fig, ax = plt.subplots(figsize=(6,6))
plt.imshow(acc)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)

print('Accuracy: %f' % (sum([acc[i,i] for i in range(num_class)]) / len(labels_val)))
print('Elapsed Time: %f s' % (time.time() - tic))

data_dir = './data'
train_list = []
val_list = []

for line in open(os.path.join(data_dir, dataset, 'train.txt'), 'r'):
    train_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))
for line in open(os.path.join(data_dir, dataset, 'val.txt'), 'r'):
    val_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))



def upload():
   
    global filename
    filename = askopenfilename(initialdir = "images")
    pathlabel.config(text=filename)
    if not os.path.exists('output'):
       os.mkdir('output')
    else:
       shutil.rmtree('output')
       os.mkdir('output')

def DCNN():
    name = os.path.basename(filename)
    arr = name.split(".")
    kdt = KDTree(features_train, leaf_size=30, metric='euclidean')
    K = 5
    q_ind = int(arr[0])
    print(features_val[q_ind:q_ind+1])
    dist, ind = kdt.query(features_val[q_ind:q_ind+1], k=K)
    print('Query image from validation set:')
    
    
    for i in range(K):
       print("distance : "+str(dist[0,i]))
       species_name.append(os.path.basename(train_list[ind[0,i]][0]));
       species_score.append((dist[0,i]/100))
       img = cv2.imread(train_list[ind[0,i]][0])
       img = cv2.resize(img,(600,300))
       cv2.imwrite("output/"+train_list[ind[0,i]][0],img)
       cv2.imshow("Predicted Species name : "+os.path.basename(train_list[ind[0,i]][0]),img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scoreGraph():
    #height = [sim1, tan]
    #bars = ('Cosine Similarity', 'Tanimoto Measure')
    y_pos = np.arange(len(species_name))
    plt.bar(y_pos, species_score)
    plt.xticks(y_pos, species_name)
    plt.show()

font = ('times', 20, 'bold')
title = Label(main, text='Bird Species Identification using Deep Learning')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Bird Image", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

depthbutton = Button(main, text="Run DCNN Algorithm & View Identified Species", command=DCNN)
depthbutton.place(x=50,y=150)
depthbutton.config(font=font1) 


viewresults = Button(main, text="View Score Graph", command=scoreGraph)
viewresults.place(x=50,y=200)
viewresults.config(font=font1) 

main.config(bg='brown')
main.mainloop()