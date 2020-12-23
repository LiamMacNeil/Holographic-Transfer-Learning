import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score, plot_precision_recall_curve, precision_recall_fscore_support, classification_report, confusion_matrix

from sklearn.preprocessing import label_binarize
import pandas as pd
from itertools import cycle

########################### GLOBAL VARIABLES ###########################
#FILE PATHS
train_path = "D:\\plankton_cultures\\Train"
test_path = "D:\\Test"
data_dir = "D:\\plankton_cultures"
save_dir = 'D:\\plankton_cultures\\saved_models'
training_data = pd.read_csv("D:\\plankton_cultures\\train_Copy.csv")
test_data = pd.read_csv("D:\\plankton_cultures\\test.csv")

#CLASS NAMES
target_names = ['Alexandrium_sp.', 'Ceratium_fus.', 'Ceratium_lin.', 'Ceratium_lon.', 'Ceratium_sp.',
                'Chaetoceros_soc.', 'Chaetoceros_sp.', 'Chaetoceros_straight', 'Crustacean', 'Distephanus_sp.',
                'Melosira_sp.', 'Noise', 'parvicorbicula_socialis', 'Prorocentrum_sp.', 'Pseudo-nitchzia_sp.',
                'Rhizosolenia_sp.', 'Rods', 'Skeletonema_sp.', 'Tintinnid']

# OBJECTS FOR MODELS AND IMAGE PARAMETERS
EPOCH = 1
k = 5
IMAGE_LENGTH = 4
CLASS_SIZE = 19
BATCH_SIZE = 32
num_of_test_samples = 4262
img_width, img_height = 128, 128
input_img = (128, 128, 3)

#KERAS CHANNELS FORMAT
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# ALLOW MEMORY GROWTH ON GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device
    pass

######################### MODEL EVALUATION METRICS #################################
# CUSTOM METRIC FUNCTIONS (ONLY USING TO GENERATE F1)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#TENSORFLOW METRICS
Metrics = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.Precision(name='precision'),
    f1_m,
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
]

######################### MODEL #################################
# PREPROCESS IMGAE TO INPUT INTO MODEL, NEEDS TO BE INSTANTIATED FOR EACH MODEL
generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Batch Norm corrections adapted from https://medium.com/towards-artificial-intelligence/batchnorm-for-transfer-learning-df17d2897db6
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Replace BatchNormalization layers with this new layer.
    This layer has fixed momentum 0.9.
    """
    def __init__(self, momentum=0.9, name=None, **kwargs):
        super(BatchNormalization, self).__init__(momentum=0.9, name=name, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=training)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config

tf.keras.layers.BatchNormalization = BatchNormalization

#MODEL FUNCTION
def Xception_model():
    input_tensor = Input(shape=(128, 128, 3))
    # Transfer learning
    Xception_model = applications.xception.Xception(weights='imagenet',
                                           include_top=False,
                                           input_tensor=input_tensor,
                                           pooling='max',
                                           layers= tf.keras.layers)

    # Freeze previous layer for first rounds of training
    for layer in Xception_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    # On top of the last max pooling layer of Inception,
    # add: flatten layer, two fully connected (fc) layers, implement dropout, softmax classifier
    last = Xception_model.layers[-1].output
    x = Flatten()(last)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.3)(x)
    x = Dense(CLASS_SIZE, activation='softmax', name='predictions')(x)
    model = Model(Xception_model.input, x)
    # Compile the model
    model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=[Metrics])
    return model

model = Xception_model()
model.summary()

######################### SKF, IMAGE GENERATOR, COMPILE AND FIT #################################
#EMPTY LISTS TO FILL FOR METRICS

#TRAINING
Train_Accuracy = []
Train_Precision = []
Train_Recall = []
Train_F1 = []
Train_auc = []
Train_fpr = []
Train_tpr = []
Train_loss = []
#VALIDATION
Val_Accuracy = []
Val_Precision = []
Val_Recall = []
Val_F1 = []
Val_auc = []
Val_Loss = []
Val_fpr = []
Val_tpr = []
#TEST
acc_scores = []
precis_scores = []
rec_scores = []
f1_scores = []
# STRATIFIED KFOLD CV, AND RANDOM STATE FOR REPRODUCIBILITY
skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
Y = training_data[['Label']]
X = training_data[['Image']]
Y_test = test_data[['Label']].values
# BINARIZE THE LABELS FOR ROC AND PRECISION-RECALL
Y_test_binarized = label_binarize(Y_test, classes=target_names)
X_test = test_data[['Image']].values

#SAVING MODEL NAME FOR EACH FOLD
def get_model_name(k):
    return 'model_' + str(k) + '.h5'

# BULKY FOR LOOP ACROSS TRAINING AND TESTING IMAGES, SPLIT INTO FOLDS,
fold_var = 1

for i, (train_index, val_index) in enumerate(skf.split(X.values, Y.values)):
    train_data = training_data.iloc[train_index]
    valid_data = training_data.iloc[val_index]

    train_data_generator = generator.flow_from_dataframe(dataframe=train_data, directory=train_path,
                                                         x_col="Image", y_col="Label", target_size=input_img[:-1],
                                                         color_mode="rgb", class_mode="categorical",
                                                         classes=target_names,
                                                         batch_size=BATCH_SIZE, shuffle=True)
    valid_data_generator = generator.flow_from_dataframe(dataframe=valid_data, directory=train_path,
                                                         x_col="Image", y_col="Label", target_size=input_img[:-1],
                                                         color_mode="rgb", class_mode="categorical",
                                                         classes=target_names,
                                                         batch_size=BATCH_SIZE, shuffle=True)
    model = Xception_model()
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Metrics])
    # Create callbacks
    # ReduceLROnPlateau: Reduce learning rate when auc stops improving for x epochs
    # ModelCheckpoint:
    # Early stopping: Cease training loop if after x epochs achieves no defined improvement
    RLRP = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="min",
        min_delta=0.0001)
    Checkpoint = ModelCheckpoint(
        filepath=save_dir + get_model_name(fold_var),
        monitor='val_loss',
        verbose=1,
        save_best_only=True, mode='min')
    Stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)
    callbacks = [
        RLRP,
        Checkpoint]
    # This saves the best model
    # FIT THE MODEL
    history = model.fit(train_data_generator,
                        epochs=EPOCH,
                        steps_per_epoch=train_data_generator.samples / BATCH_SIZE,
                        validation_data=valid_data_generator,
                        validation_steps=valid_data_generator.samples / BATCH_SIZE,
                        callbacks=callbacks,
                        verbose=2)
    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights(save_dir + "model_" + str(fold_var) + ".h5")

    #APPEND TRAINING AND VALIDATION RESULTS TO EMPTY LISTS
    Train_Accuracy.append(history.history['accuracy'])
    Train_Precision.append(history.history['precision'])
    Train_Recall.append(history.history['recall'])
    Train_F1.append(history.history['f1_m'])
    Train_fpr.append(history.history['fp'])
    Train_loss.append(history.history['loss'])

    Val_Accuracy.append(history.history['val_accuracy'])
    Val_Precision.append(history.history['val_precision'])
    Val_Recall.append(history.history['val_recall'])
    Val_F1.append(history.history['val_f1_m'])
    Val_auc.append(history.history['val_auc'])
    Val_Loss.append(history.history['val_loss'])
    Val_fpr.append(history.history['val_fp'])

    ###################### Finally, evaluating on the test set ######################
    test_generator = generator.flow_from_dataframe(dataframe=test_data, directory=test_path,
                                                   x_col="Image", y_col="Label", target_size=input_img[:-1],
                                                   color_mode="rgb", class_mode="categorical", classes=target_names,
                                                   batch_size=BATCH_SIZE, shuffle=False)
    Score = model.evaluate(x=test_generator, steps=num_of_test_samples / BATCH_SIZE, verbose=1)
    # evaluate the model

    #APPEND TEST RESULTS
    acc_scores.append(Score[1])
    rec_scores.append(Score[2])
    precis_scores.append(Score[3])
    f1_scores.append(Score[4])

    print("Fold %d:" % fold_var)
    fold_var += 1

###################### VISUALLY EVALUATING CV BEHAVIOUR ######################
# Modelled after (https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html)
epochs = list(range(EPOCH))

Train_Accuracy_mean = np.mean(Train_Accuracy, axis=0)
Train_Accuracy_std = np.std(Train_Accuracy, axis=0)
Train_Precision_mean = np.mean(Train_Precision, axis=0)
Train_Precision_std = np.std(Train_Precision, axis=0)
Train_Recall_mean = np.mean(Train_Recall, axis=0)
Train_Recall_std = np.std(Train_Recall, axis=0)
Train_F1_mean = np.mean(Train_F1, axis=0)
Train_F1_std = np.std(Train_F1, axis=0)
Train_loss_mean = np.mean(Train_loss, axis=0)
Train_loss_std = np.std(Train_loss, axis=0)

Val_Accuracy_mean = np.mean(Val_Accuracy, axis=0)
Val_Accuracy_std = np.std(Val_Accuracy, axis=0)
Val_Precision_mean = np.mean(Val_Precision, axis=0)
Val_Precision_std = np.std(Val_Precision, axis=0)
Val_Recall_mean = np.mean(Val_Recall, axis=0)
Val_Recall_std = np.std(Val_Recall, axis=0)
Val_F1_mean = np.mean(Val_F1, axis=0)
Val_F1_std = np.std(Val_F1, axis=0)
Val_loss_mean = np.mean(Val_Loss, axis=0)
Val_Loss_std = np.std(Val_Loss, axis=0)

################################# PLOTTING #################################

#PLOTTING LOG LOSS WITH ST DEV BY FOLD
plt.plot(Train_loss_mean, '-', color='b', label='Training Loss')
plt.plot(Val_loss_mean, '-', color='g', label='Validation Loss')
plt.fill_between(epochs, Train_loss_mean - Train_loss_std,
                 Train_loss_mean + Train_loss_std,
                 alpha=0.1, color='b')
plt.fill_between(epochs, Val_loss_mean - Val_Loss_std,
                 Val_loss_mean + Val_Loss_std,
                 alpha=0.1, color='g')
plt.legend(loc='upper right', fontsize = 18)
plt.title('Xception: Log Loss (Cross entropy)', fontsize= 20)
plt.xlabel('Epoch', fontsize= 18)
plt.xticks(fontsize=16)
plt.ylabel('Loss', fontsize= 18)
plt.yticks(fontsize=16)
plt.grid()
plt.show()

#PLOTTING THRESHOLD METRICS WITH ST DEV BY FOLD
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), constrained_layout=True)
axes[0, 0].plot(Train_Accuracy_mean, '-', color='b', label='Training')
axes[0, 0].plot(Val_Accuracy_mean, '-', color='g', label='Validation')
axes[0, 0].fill_between(epochs, Train_Accuracy_mean - Train_Accuracy_std,
                        Train_Accuracy_mean + Train_Accuracy_std,
                        alpha=0.1, color='b')
axes[0, 0].fill_between(epochs, Val_Accuracy_mean - Val_Accuracy_std,
                        Val_Accuracy_mean + Val_Accuracy_std,
                        alpha=0.1, color='g')

axes[0, 1].plot(Train_Precision_mean, '-', color='b', label='Training')
axes[0, 1].plot(Val_Precision_mean, '-', color='g', label='Validation')
axes[0, 1].fill_between(epochs, Train_Precision_mean - Train_Precision_std,
                        Train_Precision_mean + Train_Precision_std,
                        alpha=0.1, color='b')
axes[0, 1].fill_between(epochs, Val_Precision_mean - Val_Precision_std,
                        Val_Precision_mean + Val_Precision_std,
                        alpha=0.1, color='g')

axes[1, 0].plot(Train_Recall_mean, '-', color='b', label='Training')
axes[1, 0].plot(Val_Recall_mean, '-', color='g', label='Validation')
axes[1, 0].fill_between(epochs, Train_Recall_mean - Train_Recall_std,
                        Train_Recall_mean + Train_Recall_std,
                        alpha=0.1, color='b')
axes[1, 0].fill_between(epochs, Val_Recall_mean - Val_Recall_std,
                        Val_Recall_mean + Val_Recall_std,
                        alpha=0.1, color='g')

axes[1, 1].plot(Train_F1_mean, '-', color='b', label='Training')
axes[1, 1].plot(Val_F1_mean, '-', color='g', label='Validation')
axes[1, 1].fill_between(epochs, Train_F1_mean - Train_F1_std,
                        Train_F1_mean + Train_F1_std,
                        alpha=0.1, color='b')
axes[1, 1].fill_between(epochs, Val_F1_mean - Val_F1_std,
                        Val_F1_mean + Val_F1_std,
                        alpha=0.1, color='g')
axes[0, 0].set_title("Accuracy", fontsize= 20)
axes[0, 0].set_ylabel('Score', fontsize= 16)
axes[0, 0].set_xlabel('Epoch', fontsize= 16)
axes[0, 1].set_title("Precision", fontsize= 20)
axes[0, 1].set_ylabel('Score', fontsize= 16)
axes[0, 1].set_xlabel('Epoch', fontsize= 16)
axes[1, 0].set_title("Recall", fontsize= 20)
axes[1, 0].set_ylabel('Score', fontsize= 16)
axes[1, 0].set_xlabel('Epoch', fontsize= 16)
axes[1, 1].set_title("F1-Score", fontsize= 20)
axes[1, 1].set_ylabel('Score', fontsize= 16)
axes[1, 1].set_xlabel('Epoch', fontsize= 16)
axes[0, 0].set_xlim(0, EPOCH - 1)
axes[0, 0].set_ylim(0.5, 1.01)
axes[0, 0].grid()
axes[0, 1].set_xlim(0, EPOCH - 1)
axes[0, 1].set_ylim(0.5, 1.01)
axes[0, 1].grid()
axes[1, 0].set_xlim(0, EPOCH - 1)
axes[1, 0].set_ylim(0.5, 1.01)
axes[1, 0].grid()
axes[1, 1].set_xlim(0, EPOCH - 1)
axes[1, 1].set_ylim(0.5, 1.01)
axes[1, 1].grid()

plt.suptitle('Xception', fontsize = 22)
plt.legend(loc='lower right', fontsize = 18)
plt.show()

###################### TEST SET RESULTS ######################
print(model.metrics_names)
print(Score)
print('Acc Scores Mean: %.3f, Standard Deviation: %.3f' % (np.mean(acc_scores, axis=0), np.std(acc_scores, axis=0)))
print('Rec Scores Mean: %.3f, Standard Deviation: %.3f' % (np.mean(rec_scores, axis=0), np.std(rec_scores, axis=0)))
print('Prec Scores Mean: %.3f, Standard Deviation: %.3f' % (np.mean(precis_scores, axis=0), np.std(precis_scores, axis=0)))
print('F1 Scores Mean: %.3f, Standard Deviation: %.3f' % (np.mean(f1_scores, axis=0), np.std(f1_scores, axis=0)))

Final_Score = model.evaluate(x=test_generator, steps=num_of_test_samples / BATCH_SIZE, verbose=1)

y_pred = model.predict(x=test_generator, steps=num_of_test_samples / BATCH_SIZE, verbose=1)
print("%.3f%% (+/- %.3f%%)" % (np.mean(Final_Score[1])*100, np.std(Final_Score[1])*100))
print("%.3f%% (+/- %.3f%%)" % (np.mean(Final_Score[2])*100, np.std(Final_Score[2])*100))
print("%.3f%% (+/- %.3f%%)" % (np.mean(Final_Score[3])*100, np.std(Final_Score[3])*100))
print("%.3f%% (+/- %.3f%%)" % (np.mean(Final_Score[4])*100, np.std(Final_Score[4])*100))


############################## PRECISION-RECALL CURVES ##############################
# Plot linewidth.
lw = 2
colors = cycle(['rosybrown', 'lightcoral', 'darkred', 'red',
                'darkorange', 'darkgoldenrod', 'gold', 'olive',
                'olivedrab', 'greenyellow', 'forestgreen', 'aquamarine',
                'aqua', 'deepskyblue', 'lightsteelblue', 'blue',
                'blueviolet', 'plum', 'hotpink'])

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(CLASS_SIZE):
    precision[i], recall[i], _ = precision_recall_curve(Y_test_binarized[:, i],
                                                        y_pred[:, i])
    average_precision[i] = average_precision_score(Y_test_binarized[:, i], y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test_binarized.ravel(),
    y_pred.ravel())
average_precision["micro"] = average_precision_score(Y_test_binarized, y_pred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure()
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(CLASS_SIZE), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append(target_names[i]+'(area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize = 18)
plt.xticks(fontsize=16)
plt.ylabel('Precision', fontsize = 18)
plt.yticks(fontsize=16)
plt.title('Xception', fontsize= 20)
plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), prop=dict(size=12), fancybox=True, shadow= True, ncol=5)

plt.show()