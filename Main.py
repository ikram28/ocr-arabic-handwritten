from untitled1 import *
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import RandomState
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import string
from shutil import copyfile, rmtree
import re
import cv2
from PIL import Image, ImageDraw
import glob
from keras.callbacks import EarlyStopping


save_path = "C:/Users/ASUS ROG STRIX/Desktop/OCR"
path = "C:/Users/ASUS ROG STRIX/Desktop/OCR/enit_ifn database/ifnenit-database-demo/data/set_a/tif"
model_name = "OCR_IFNENIT_verl"

prng = RandomState(32)


batch_size = 64
imgh = 100
imgw = 300

try:
    rmtree(save_path + "/" + model_name)
except:
    pass

os.mkdir(save_path + "/" + model_name)

train = [dp + "/" + f for dp, dn, filenames in os.walk(path)
         for f in filenames if re.search('tif', f)]

prng.shuffle(train)
lexicon = get_lexicon_2(train)
classes = {j: i for i, j in enumerate(lexicon)}
inve_classes = {v: k for k, v in classes.items()}

length = len(train)
train, val = train[:int(length * 0.9)], train[int(length * 0.9):]
lenghts = get_lengths(train)
max_len = max(lenghts.values())

objet = Readf(classes=classes)



img_w, img_h = 300, 100
output_size = len(classes) +1
crnn = CRNN(img_w, img_h, output_size, max_len)
model = crnn.model

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

train_generator = objet.run_generator(train)
val_generator = objet.run_generator(val)

train_steps = len(train) // batch_size
val_steps = len(val) // batch_size + 1

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=train_steps,
                              validation_data=val_generator,
                              validation_steps=val_steps,
                              epochs=150)
                              #callbacks=[early_stopping])




# Create empty arrays to store the evaluation results
losses = []

# Iterate over the validation generator and calculate loss for each batch
for i in range(val_steps):
    inputs, targets = next(val_generator)
    loss = model.evaluate(inputs, targets, verbose=0)
    losses.append(loss)

# Calculate the average validation loss
validation_loss = np.mean(losses)

# Print the evaluation result
print("Validation Loss:", validation_loss)



# Call the plot_training_history function
def plot_training_history(history):
    import matplotlib.pyplot as plt
    
   
    plt.figure(figsize=(12, 6))
    # A dictionary containing the recorded values of different metrics during training
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

plot_training_history(history)


#######################################################################################################################################################################



def num_to_label(num, inv_classes):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            char = inv_classes.get(ch)
            if char is not None:  # Check if the character exists in the mapping
                ret += char
    return ret



# Create empty arrays to store the predictions
predictions_ctc_loss = []
predictions_probabilities = []

# Iterate over the validation generator and make predictions for each batch
for i in range(val_steps):
    inputs, _ = next(val_generator)

    # Reshape the input images to match the expected shape of the model
    inputs['the_input'] = inputs['the_input'].reshape((-1, imgw, imgh))

    # Make predictions with the model
    batch_ctc_loss, batch_predictions = model.predict(inputs)

    predictions_ctc_loss.append(batch_ctc_loss)
    predictions_probabilities.append(batch_predictions)

# Concatenate the predictions for all batches
predictions_ctc_loss = np.concatenate(predictions_ctc_loss, axis=0)
predictions_probabilities = np.concatenate(predictions_probabilities, axis=0)


#Best-path decoding
decoded = K.get_value(K.ctc_decode(predictions_probabilities, input_length=np.ones(predictions_probabilities.shape[0])*predictions_probabilities.shape[1], greedy=False)[0][0])

decoded_words = []
for d in decoded:
    decoded_word = num_to_label(d, inve_classes)
    decoded_words.append(decoded_word)
    

true_words = []  # List to store the true words

# Iterate over the validation set
for image_path in val:
    true_word = evaluate_word(image_path)  # Get the true word for the current image
    true_words.append(true_word)
# Compare the true words with the decoded words
for true_word, decoded_word in zip(true_words, decoded_words):
    if true_word == decoded_word:
        print("Correct: ", true_word)
    else:
        print("Incorrect: True word =", true_word, ", Decoded word =", decoded_word)  

