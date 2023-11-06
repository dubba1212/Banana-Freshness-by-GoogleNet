from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras.models import model_from_json
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses, activations, models
import cv2
from keras.preprocessing import image

main = tkinter.Tk()
main.title("Monitoring the Change Process of Banana Freshness by GoogLeNet") #designing main screen
main.geometry("1300x1200")

global filename
global model

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");

def loadModel():
    global model
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"GoogLeNet Trasfer Learning Model Prediction Accuracy = "+str(accuracy)+"\n\n")
        text.insert(END,"See Black Console to view GoogLNet layers\n")
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,
                                           shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1.0/255.)
        #reading images from dataset/train folder
        train_generator = train_datagen.flow_from_directory('Dataset/train', batch_size = 20, class_mode = 'categorical', target_size = (150, 150))
        validation_generator = test_datagen.flow_from_directory('Dataset/train', batch_size = 20, class_mode = 'categorical', target_size = (150, 150))
        #creating google net inceptionv3 model by ignoring its top model details and using imagenet weight
        #the object name is base_model
        base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')
        #last google net model layer will be ignore to concatenate banana custom model
        base_model.trainable = False
        #getting 3 class from banana dataset as ripe, over ripe and green
        print(train_generator.class_indices)
        #creating own model object
        add_model = Sequential()
        #adding google net base_model object to our custome model
        add_model.add(base_model)
        add_model.add(GlobalAveragePooling2D())
        add_model.add(Dropout(0.5))
        add_model.add(Dense(3, activation='softmax'))

        model = add_model
        #compiling model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        #start training model
        hist = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
        model.save_weights('model/model_weights.h5')
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"GoogLeNet Trasfer Learning Model Prediction Accuracy = "+str(accuracy)+"\n\n")
        text.insert(END,"See Black Console to view GoogLNet layers\n")

def predictChange():
    classes = ['Green','Over Ripe','Ripe']
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (150,150))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,150,150,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    '''
    imagetest = image.load_img(filename, target_size = (150,150))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis = 0)
    '''
    preds = model.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Banana Changes Predicted As : '+classes[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Banana Changes Predicted As : '+classes[predict], img)
    cv2.waitKey(0)
    


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('GoogLeNet Accuracy & Loss Graph')
    plt.show()

def close():
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Monitoring the Change Process of Banana Freshness by GoogLeNet')
title.config(bg='goldenrod2', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Banana Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Generate & Load GoogLeNet Model", command=loadModel, bg='#ffb3fe')
modelButton.place(x=270,y=550)
modelButton.config(font=font1) 

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=590,y=550)
graphButton.config(font=font1) 

predictButton = Button(main, text="Upload Banana Test Image & Monitor Change", command=predictChange, bg='#ffb3fe')
predictButton.place(x=810,y=550)
predictButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
exitButton.place(x=1190,y=550)
exitButton.config(font=font1) 


main.config(bg='SpringGreen2')
main.mainloop()
