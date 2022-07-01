
import os
import wget
import zipfile
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from utils import angle_error, RotNetDataGenerator

filepath = r'images'

def get_filenames(path):

    list = os.listdir(path)
    image_paths = []
    for filename in list:
        view_id = filename.split('.')[0]
        image_paths.append(os.path.join(path, filename))
    print(image_paths)

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.9)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    return train_filenames, test_filenames

train_filenames, test_filenames = get_filenames(filepath)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'rotnet_resnet50_x'

# number of classes
nb_classes = 360
# input image shape
input_shape = (350, 350, 3)

# load base model
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)

# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01, momentum=0.9),
              metrics=[angle_error])

# training parameters
batch_size = 5
nb_epoch = 5

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)
