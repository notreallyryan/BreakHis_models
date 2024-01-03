from keras.applications import MobileNetV2, mobilenet_v2
from keras import Input
from keras.applications import ResNet50V2, resnet_v2
from keras.applications import VGG19, vgg19
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, MaxPool2D, ReLU, Conv2D, Flatten, Resizing
from keras.layers import RandomFlip, RandomRotation, RandomHeight, RandomWidth, RandomZoom, Normalization
from keras_cv.layers import RandomShear
from keras import Model, Sequential
from keras import backend as K

"""
Data Augmenter - add some variation to the dataset to reduce overfitting

values taken from Agarwal et al.
"""
def augmenter():
    ROTATION = 0.7 #dis in radians, more accurately, Agarwal uses 40 degrees (which is 0.694 radians). Unfortunately, I am not Agarwal.
    HEIGHT = 0.2
    WIDTH = 0.2
    SHEAR = 0.2
    ZOOM = 0.2

    data_augmentation = Sequential()

    #flip
    data_augmentation.add(RandomFlip())
    #rotation
    data_augmentation.add(RandomRotation(ROTATION))
    #width
    data_augmentation.add(RandomWidth(WIDTH))
    #height
    data_augmentation.add(RandomHeight(HEIGHT))
    #shear
    data_augmentation.add(RandomShear(SHEAR))
    #zoom
    data_augmentation.add(RandomZoom(ZOOM))

    return data_augmentation


"""
uses mobilenetv2 as the base, retrains final fully connected layer
"""
def MobileNetV2_base(imgsize): #it's not stealing if it's transfer learning!
    DROPOUT_RATIO = 0.5
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy', precision, recall, f1]
    RETRAIN_LAYERS = 5

    base = MobileNetV2(input_shape= (224,224,3),include_top=False, weights='imagenet')

    #change accordingly
    fine_tune_at = len(base.layers)-RETRAIN_LAYERS
    #freeze! don't move!
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    #new input layer
    inputs = Input(shape=imgsize)

    #augmentations
    x = augmenter()(inputs)

    #normalizations
    x = Normalization()(x)

    #resize inputs to match MobileNetV2 architecture
    x = Resizing(height = 224, width= 224)(x)

    #preprocess using same weights model was trained on:
    x = mobilenet_v2.preprocess_input(x)

    #transfer learning layers
    x = base(x, training= False)

    #new layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATIO)(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer= OPTIMIZER, loss = LOSS, metrics = METRICS)

    return model



"""
uses ResNet50v2 as the base, retrains final fully connected layer
"""
def ResNet50v2_base(imgsize): #okay maybe it's stealing.
    DROPOUT_RATIO = 0.5
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy', precision, recall, f1]
    RETRAIN_LAYERS = 5

    base = ResNet50V2(input_shape = (224,224,3), include_top = False, weights = 'imagenet')
    
    #change accordingly
    fine_tune_at = len(base.layers)-RETRAIN_LAYERS

    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = Input(shape=imgsize)

    #augmentations
    x = augmenter()(inputs)

    #normalizations
    x = Normalization()(x)

    #reshaping for transfer learning
    x = Resizing(height = 224, width= 224)(x)

    #preprocessing used by base
    x = resnet_v2.preprocess_input(x)

    #old
    x = base(x, training = False)

    #new
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATIO)(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer= OPTIMIZER, loss = LOSS, metrics = METRICS)

    return model



"""
uses VGG19 as the base, retrains final fully connected layer
"""
def VGG19_base(imgsize):
    DROPOUT_RATIO = 0.5
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy', precision, recall, f1]
    RETRAIN_LAYERS = 5

    base = VGG19(input_shape = (224,224,3), include_top = False, weights = 'imagenet')

    #change accordingly
    fine_tune_at = len(base.layers)-RETRAIN_LAYERS

    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = Input(shape=imgsize)

    #augmentations
    x = augmenter()(inputs)

    #normalizations
    x = Normalization()(x)

    x = Resizing(height = 224, width= 224)(x)

    #preprocessing used by base
    x = vgg19.preprocess_input(x)

    #old
    x = base(x, training = False)

    #new
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATIO)(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer= OPTIMIZER, loss = LOSS, metrics = METRICS)

    return model



"""
Custom model, adapted from Agarwal et al.
DOI: 10.1007/978-981-16-2641-8_8
"""

def custom_model(imgsize):
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy', precision, recall, f1]

    inputs= Input(shape=imgsize)

    x = augmenter()(inputs)

    #normalizations
    x = Normalization()(x)

    #reshaping
    x = Resizing(height = 224, width= 224)(x)

    #L1
    Z1 = Conv2D(filters = 32, kernel_size = (3,3), strides = 1, padding = 'SAME')(x)
    A1 = ReLU()(Z1)
    #L2
    P1 = MaxPool2D(pool_size = (2,2), strides = 1, padding = 'SAME')(A1)
    #L3
    Z2 = Conv2D(filters = 32, kernel_size = (3,3), strides = 1, padding = 'SAME')(P1)
    A2 = ReLU()(Z2)
    #L4
    P2 = MaxPool2D(pool_size = (2,2), strides = 1, padding = 'SAME')(A2)
    #L5
    Z3 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'SAME')(P2)
    A3 = ReLU()(Z3)
    #L6
    P3 = MaxPool2D(pool_size = (2,2), strides = 1, padding = 'SAME')(A3)
    #Flattten
    F = Flatten()(P3)
    #FCL1
    fcl1 = Dense(64, activation = 'relu')(F)
    #FCL2
    fcl2 = Dense(1, activation = 'sigmoid')(fcl1)
    model = Model(inputs=inputs, outputs=fcl2)
    model.compile(optimizer= OPTIMIZER, loss = LOSS, metrics = METRICS)
    
    return model

"""
metric functions are below, taken from https://neptune.ai/blog/keras-metrics 
"""

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    r = recall(y_true, y_pred)
    p= precision(y_true, y_pred)
    f1 = 2*((p*r)/(p+r+K.epsilon()))
    return f1



#testing stuff
if __name__ == "__main__":
    #MobileNetV2_base(460, 700, 3)
    #ResNet50v2_base(460, 700, 3)
    pass
