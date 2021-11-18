#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Projet KAGGLE : SVM et réseaux de neuronnes S1 - M2 EKAP 
# Gabin LAGRE et Astrid VALICON

# Tâche : déterminer quand un signal est présent dans les données (target = 1)
# prévoir la probabilité que l’observation contienne une onde gravitationnelle


# In[19]:


# Librairies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from glob import glob # importation des données .npy
from tqdm import tqdm # graphiques
import librosa 
import librosa.display
from sklearn import datasets
import tensorflow as tf

# Pour l'estimation du modèle
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten

# Évaluation de notre modèle
from tensorflow.keras.metrics import AUC

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC

import librosa
import torch

from nnAudio.Spectrogram import CQT1992v2
import efficientnet.keras as efn


# In[2]:


train_label = pd.read_csv('/Users/valiconastrid/Desktop/EKAP S3/SVM et réseaux de neurones/Kaggle/training_labels.csv', header=0)
test = pd.read_csv('/Users/valiconastrid/Desktop/EKAP S3/SVM et réseaux de neurones/Kaggle/sample_submission.csv', header=0)


# In[3]:


print(f'Training labels: {train_label.shape[0]} | Test dataset: {test.shape[0]}')


# In[4]:


train_label.describe()


# In[24]:


sns.countplot(data=train_label, x="target", palette="Set3", )


# In[4]:


train_label['target'].value_counts()


# In[5]:


train_label.isnull().sum() # pas de valeurs nulles


# In[5]:


train_label.head


# In[3]:


train_path = glob('/Users/valiconastrid/Desktop/EKAP S3/SVM et réseaux de neurones/Kaggle/train/*/*/*/*')


# In[4]:


len(train_path)


# In[9]:


explore_sample_3 = np.load(train_path[3])
explore_sample_3


# In[10]:


explore_sample_2 = np.load(train_path[2])
explore_sample_2


# In[11]:


explore_sample_3.shape


# In[38]:


explore_sample_2.shape


# In[6]:


print(len(explore_sample_3[0]), len(explore_sample_3[1]), len(explore_sample_3[2]))


# In[12]:


train_path[3]


# In[13]:


rind = train_path[3].rindex('/') # last index where the character '/' appeared
extracted_id_for_explore_sample_3 = train_path[3][rind+1:].replace('.npy', '') # replaced .npy
extracted_id_for_explore_sample_3


# In[14]:


train_label[train_label['id']==extracted_id_for_explore_sample_3]['target']


# In[15]:


positive_sample = explore_sample_3
negative_sample = np.load(train_path[1])
negative_sample


# In[16]:


samples = (positive_sample, negative_sample)
targets = (1, 0)


# In[17]:


colors = ("red", "green", "blue")
signal_names = ("LIGO Hanford", "LIGO Livingston", "Virgo")

for x, i in tqdm(zip(samples, targets)):
    figure = plt.figure(figsize=(16, 7))
    figure.suptitle(f'Raw wave (target={i})', fontsize=20)
    # range is 3 because we have 3 different rows for each interferometers
    for j in range(3):
        axes = figure.add_subplot(3, 1, j+1)
        librosa.display.waveshow(x[j], sr=2048, ax=axes, color=colors[j])
        axes.set_title(signal_names[j], fontsize=12)
        axes.set_xlabel('Time[sec]')
    plt.tight_layout()
    plt.show()


# In[18]:


sns.distplot(positive_sample[0,:])


# In[5]:


pd.set_option('display.max_colwidth',None)


# In[6]:


ids = []
for files in train_path:
    ids.append(files[files.rindex('/')+1:].replace('.npy',''))
df = pd.DataFrame({"id":ids,"path":train_path})
df = pd.merge(df, train_label, on='id')


# In[21]:


df.head()


# In[9]:


#df.to_csv('tableau_kaggle.csv')


# In[22]:


train_label.shape


# In[11]:


df.shape


# In[78]:


df


# In[7]:


# Nous définissons ici les différents paramètres des signaux 
sample_rate = 2048 # Les données fournies sont à 2048 Hz
signal_length = 2 # Chaque signal dure 2 s
fmin, fmax = 20, 500 # Nous mettons un filtre de minimum 20 Hz et maximum 500 Hz. 
hop_length = 64 # Paramètre de longueur de saut. 

# Paramètre du modèle
batch_size = 250 # nombre d'images utilisées pour entraîner le réseau
epochs = 3 # nombre d'epochs : correspond à un apprentissage sur toutes les données, plus ce nombre est grand plus on devrait obtenir une bonne précision, mais c'est plus c’est long


# In[9]:


def get_npy_filepath(id_, is_train=True):
    path = ''
    if is_train:
        return f'/Users/valiconastrid/Desktop/EKAP S3/SVM et réseaux de neurones/Kaggle/train/{id_[0]}/{id_[1]}/{id_[2]}/{id_}.npy'
    else:
        return f'/Users/valiconastrid/Desktop/EKAP S3/SVM et réseaux de neurones/Kaggle/test/{id_[0]}/{id_[1]}/{id_[2]}/{id_}.npy'


# In[8]:


# Nous définissons la transformation Q-Constant
cq_transform = CQT1992v2(sr=sample_rate, fmin=fmin, fmax=fmax, hop_length=hop_length)


# In[10]:


# fonction pour charger le fichier et récupérer l'image (spectogramme).
def parse_function(id_path):
    # télécharger le fichier .npy
    signals = np.load(id_path.numpy())
    
    # boucle à travers chaque signal 
    for i in range(signals.shape[0]):
        # normaliser les données du signal
        signals[i] /= np.max(signals[i])
    
    # empiler les tableaux en un seul vecteur
    signals = np.hstack(signals)
    
    # convertir les signaux en torch.tensor pour passer en CQT
    signals = torch.from_numpy(signals).float()
    
    # récupérer le CQT
    image = cq_transform(signals)
    
    # convertir l’image de torch.tensor en réseau
    image = np.array(image)
    
    # transposer l’image pour obtenir la bonne orientation
    image = np.transpose(image,(1,2,0))
    
    # convertir l’image en tf.tensor
    return tf.convert_to_tensor(image)


# In[11]:


# voir l'image et la taille 
image = parse_function(tf.convert_to_tensor(df['path'][1]))
print(image.shape)


# In[14]:


input_shape = (56, 193, 1)


# In[13]:


X = df['id']
y = df['target'].astype('int8').values


# In[15]:


# Diviser les ID de formation en ensembles de données de formation et de validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, stratify=y)

# Attribution des ID de test
X_test = test[['id']]


# In[16]:


# fonction TF 
def tf_parse_function(id_path, y=None):
    
    [x] = tf.py_function(func=parse_function, inp=[id_path], Tout=[tf.float32])
    x = tf.ensure_shape(x, input_shape)
    
    # si train/valid alors on retourne x, y; et pour le test, on retourne seulement x
    if y is None:
        return x
    else:
        return x, y


# In[37]:


print(y_train)


# In[58]:


X_train = pd.DataFrame(X_train)
X_valid = pd.DataFrame(X_valid)
X_test = pd.DataFrame(X_test)
print(X_train)


# In[22]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train.apply(get_npy_filepath).values, y_train))
# shuffle the dataset
train_dataset = train_dataset.shuffle(len(X_train))
train_dataset = train_dataset.map(tf_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


# In[23]:


# valid dataset

valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid.apply(get_npy_filepath).values, y_valid))
valid_dataset = valid_dataset.map(tf_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)


# In[24]:


# test dataset

test_dataset = tf.data.Dataset.from_tensor_slices((X_test['id'].apply(get_npy_filepath, is_train=False).values))

test_dataset = test_dataset.map(tf_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(batch_size)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# In[25]:


# Initiation du modèle séquentiel 
model_cnn = Sequential(name='CNN_model')

# On ajoute la premiere couche Convoluted2D puis input_shape & MaxPooling2D
model_cnn.add(Conv2D(filters=16,
                     kernel_size=3,
                     input_shape=input_shape,
                     activation='relu',
                     name='Conv_01'))
model_cnn.add(MaxPooling2D(pool_size=2, name='Pool_01'))

# Seconde couche Conv1D et MaxPooling1D
model_cnn.add(Conv2D(filters=32,
                     kernel_size=3,
                     input_shape=input_shape,
                     activation='relu',
                     name='Conv_02'))
model_cnn.add(MaxPooling2D(pool_size=2, name='Pool_02'))

# troisième couche Conv1D et MaxPooling1D
model_cnn.add(Conv2D(filters=64,
                     kernel_size=3,
                     input_shape=input_shape,
                     activation='relu',
                     name='Conv_03'))
model_cnn.add(MaxPooling2D(pool_size=2, name='Pool_03'))

# Couche applati (flatten layer)
model_cnn.add(Flatten(name='Flatten'))

# Ajout de la couche dense (dense layer)
model_cnn.add(Dense(units=512,
                activation='relu',
                name='Dense_01'))
model_cnn.add(Dense(units=64,
                activation='relu',
                name='Dense_02'))

# ajout de la couche finale avec fonction d'activation
model_cnn.add(Dense(1, activation='sigmoid', name='Output'))


# In[26]:


# Modèle CNN 
model_cnn.summary()


# In[27]:


# Évaluation du modèle CNN
model_cnn.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=[[AUC(), 'accuracy']])


# In[28]:


history_cnn = model_cnn.fit(x=train_dataset,
                            epochs=epochs,
                            validation_data=test_dataset,
                            batch_size=batch_size,
                            verbose=1)


# In[29]:


# re-train the model on remaining validation data
model_cnn.fit(x=valid_dataset, epochs=epochs, batch_size=batch_size, verbose=1)


# In[ ]:


# Prédictions
preds_cnn = model_cnn.predict(test_dataset)


# In[ ]:


#On sauvegarde
get_kaggle_format(preds_cnn, model='cnn')


# In[ ]:


submission = pd.DataFrame({'id': x_test.id, 'target': preds_cnn})


# In[ ]:


submission.to_csv('C:/Users/gabin/OneDrive/Bureau/kaggle/submission.csv', index= False)


# In[ ]:


submission=pd.read_csv("C:/Users/gabin/OneDrive/Bureau/kaggle/submission.csv", header=0)


# In[ ]:


submission.head() 

