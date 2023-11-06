''''
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
import pickle


train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory('data/train', batch_size = 20, class_mode = 'binary', target_size = (150, 150))
validation_generator = test_datagen.flow_from_directory('data/train', batch_size = 20, class_mode = 'binary', target_size = (150, 150))
base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    layers = layer

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])
inc_history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)

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
print("Training Model Accuracy = "+str(accuracy))
'''


from keras.models import Sequential
from keras.models import Model
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras.models import model_from_json
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses, activations, models

train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory('Dataset/train', batch_size = 20, class_mode = 'categorical', target_size = (150, 150))
validation_generator = test_datagen.flow_from_directory('Dataset/train', batch_size = 20, class_mode = 'categorical', target_size = (150, 150))
base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')
base_model.trainable = False
print(train_generator.class_indices)

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(3, activation='softmax'))

model = add_model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

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
print("Training Model Accuracy = "+str(accuracy))















