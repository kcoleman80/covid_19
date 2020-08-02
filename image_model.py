#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# !pip install --upgrade pip

# !pip install https://github.com/raghakot/keras-vis/archive/master.zip

# In[2]:


get_ipython().system('pip install --upgrade tensorflow')


# In[3]:


get_ipython().system('pip install --upgrade keras')


# In[4]:


get_ipython().system('pip help install --use-feature=2020-resolver')


# In[5]:


get_ipython().system('pip install git+https://github.com/keras-team/keras-preprocessing.git')


# !pip install opencv-python

# !pip install sklearn

# !pip install -U git+https://github.com/yhat/ggpy.git@v0.6.6

# # Libraries needed for the tutorial
# import pandas as pd
# import numpy as np 
# import requests
# import io
# import sklearn as scikit_learn
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# 
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.densenet import DenseNet121
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.models import Model
# from keras import backend as K
# 
# from keras.models import load_model
# 

# In[21]:


# Libraries needed for the tutorial
import pandas as pd
import numpy as np 
import requests
import io
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


# # Downloading the csv file from your GitHub account
# 
# url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv" # Make sure the url is the raw version of the file on GitHub
# download = requests.get(url).content
# 
# # Reading the downloaded content and turning it into a pandas dataframe
# 
# df = pd.read_csv(io.StringIO(download.decode('utf-8')))
# 
# # Printing out the first 5 rows of the dataframe
# 
# print (df.head())

# In[22]:


import torchxrayvision as xrv


# In[23]:


import torchvision, torchvision.transforms
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])


# In[24]:


nn_model = xrv.models.DenseNet(weights="all")#.cuda()
nn_model.op_threshs = None # to prevent sigmoid


# In[25]:


#Just tring out a binary file below
#df = pd.read_csv('mod_covid.csv')
#print(df.head())


# # Data Augmentation to improve classification of only 800 or so X-rays

# from keras_preprocessing.image import ImageDataGenerator

# # load all images in a directory
# from os import listdir
# from matplotlib import image
# # load all images in a directory
# loaded_images = list()
# for filename in listdir('/Users/kellymclean/COVID_19_Lungs/images/'):
#     # load image
#     img_data = image.imread('/Users/kellymclean/COVID_19_Lungs/images/' + filename)
#     # store loaded image
#     loaded_images.append(img_data)
#     print('> loaded %s %s' % (filename, img_data.shape))

# # DATA CLEANING

# In[26]:


df = pd.read_csv('mod_covid_monday.csv')
print(df.head())


# In[27]:


list(df.columns.values)


# In[28]:


#Medical diagnosis 
labels = ['Accelerated Phase Usual Interstitial Pneumonia',
 'Allergic bronchopulmonary aspergillosis',
 'ARDS',
 'Aspiration pneumonia',
 'Bacterial',
 'Chlamydophila',
 'Chronic eosinophilic pneumonia',
 'COVID-19',
 'COVID-19 and ARDS',
 'Cryptogenic Organizing Pneumonia',
 'Desquamative Interstitial Pneumonia',
 'E.Coli',
 'Eosinophilic Pneumonia',
 'Herpes pneumonia',
 'Influenza',
 'Invasive Aspergillosis',
 'Klebsiella',
 'Legionella',
 'Lipoid',
 'Lobar Pneumonia',
 'Lymphocytic Interstitial Pneumonia',
 'Multilobar Pneumonia',
 'Mycoplasma Bacterial Pneumonia',
 'No Finding',
 'Nocardia',
 'Organizing Pneumonia',
 'Pneumocystis',
 'Pneumonia',
 'Round pneumonia',
 'SARS',
 'Spinal Tuberculosis',
 'Streptococcus',
 'Swine-Origin Influenza A (H1N1) Viral Pneumonia',
 'todo',
 'Tuberculosis',
 'Unknown',
 'Unusual Interstitial Pneumonia',
 'Varicella']           


# In[29]:


print(labels)


# In[30]:


len(df.columns)


# In[31]:


cols = df.columns[:40] # first 40 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))


# In[32]:


# if it's a larger dataset and the visualization takes too long can do this.
# % of missing.
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# # Label train and test images using this code

# In[33]:


df.info()


# In[34]:


# Get column data types
df.dtypes
# Also check if the column is unique
for i in df:
  print('{} is unique: {}'.format(i, df[i].is_unique))


# In[35]:


#See if this # fixes model complie
#df.dtypes[df.dtypes != 'int64'][df.dtypes != 'float64']


# In[36]:


# Get column names
column_names = df.columns
print(column_names)


# In[37]:


# Check the index values
df.index.values


# In[38]:


print(len(df))


# In[39]:


print(df.shape)


# In[40]:


print(labels)


# In[41]:


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


# In[42]:


train, validate, test = train_validate_test_split(df)


# In[43]:


train.to_csv('train.csv')


# In[44]:


validate.to_csv('validate.csv')


# In[45]:


test.to_csv('test.csv')


# In[46]:


print(train.head())


# In[47]:


train.shape


# In[48]:


print(test.head())


# In[49]:


test.shape


# In[50]:


print(validate.head())


# In[51]:


validate.shape


# In[52]:


train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('validate.csv')

test_df = pd.read_csv('test.csv')

test_df.head()


# In[53]:


def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    
         
    df1_patients_unique = set(df1[patient_col].unique().tolist())
    df2_patients_unique = set(df2[patient_col].unique().tolist())
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) >= 1 # boolean (true if there is at least 1 patient in both groups)
   
    
    return leakage


# In[54]:


# test
print("test case 1")
df1 = pd.DataFrame({'patientid': [0, 1, 2]})
df2 = pd.DataFrame({'patientid': [2, 3, 4]})
print("df1")
print(df1)
print("df2")
print(df2)
print(f"leakage output: {check_for_leakage(df1, df2, 'patientid')}")
print("-------------------------------------")
print("test case 2")
df1 = pd.DataFrame({'patientid': [0, 1, 2]})
df2 = pd.DataFrame({'patientid': [3, 4, 5]})
print("df1:")
print(df1)
print("df2:")
print(df2)

print(f"leakage output: {check_for_leakage(df1, df2, 'patientid')}")


# In[55]:


print("leakage between train and test: {}".format(check_for_leakage(train, test, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(validate, test, 'PatientId')))


# In[56]:


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """
                                                                    
    
    
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


# In[57]:


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# import os
# import numpy as np
# import matplotlib.pyplot as mpplot
# import matplotlib.image as mpimg
# 
# images = []
# path = './Users/kellymclean/COVID_19_Lungs/images/'
# 
# for root, _, files in os.walk(path):
#     current_directory_path = os.path.abspath(root)
#     for f in files:
#         name, ext = os.path.splitext(f)
#         if ext == '.png' or 'jpeg' or 'jpg':
#             current_image_path = os.path.join(current_directory_path, f)
#             current_image = mpimg.imread(current_image_path)
#             images.append(current_image)
# 
# for img in images:
#     print(img.shape)

# In[58]:


IMAGE_DIR = '/Users/kellymclean/COVID_19_Lungs/images'
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# In[59]:


x, y = train_generator.__getitem__(0)
print(x.shape)
print(y.shape)
plt.imshow(x[2]);
#plt.imshow(x[1]);


# In[60]:


x, y = train_generator.__getitem__(0)
print(x.shape)
print(y.shape)
plt.imshow(x[2]);


# # Addressing Class Imbalance

# In[61]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# In[62]:



def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / labels.shape[0]
    negative_frequencies = 1 - positive_frequencies

  
    return positive_frequencies, negative_frequencies


# In[63]:


# Test
labels_matrix = np.array(
    [[1, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 1],
     [1, 0, 1]]
)
print("labels:")
print(labels_matrix)

test_pos_freqs, test_neg_freqs = compute_class_freqs(labels_matrix)

print(f"pos freqs: {test_pos_freqs}")

print(f"neg freqs: {test_neg_freqs}")


# In[64]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos


# In[65]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


# In[66]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# In[67]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);


# In[68]:


#Positive/Negative frequences for eachc lass
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # N = total number of patients (rows)

    N = labels.shape[0]

    positive_frequencies = np.sum(labels==1, axis=0) / labels.shape[0]
    negative_frequencies = 1 - positive_frequencies

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies
  


# In[69]:


# Test
labels_matrix = np.array(
    [[1, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 1],
     [1, 0, 1]]
)
print("labels:")
print(labels_matrix)

test_pos_freqs, test_neg_freqs = compute_class_freqs(labels_matrix)

print(f"pos freqs: {test_pos_freqs}")

print(f"neg freqs: {test_neg_freqs}")


# In[70]:


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    
    def weighted_loss(y_true, y_pred):
        
        # initialize loss to zero
        loss = 0.0
        

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += K.mean(-(pos_weights[i]*y_true[:, i]*K.log(y_pred[:, i]+epsilon)
                             + neg_weights[i]*(1-y_true[:, i])*K.log((1-y_pred[:, i])+epsilon)))
        return loss
    
    return weighted_loss


# # Test
# import numpy as np
# import tensorflow.python.keras.backend as K
# sess = K.get_session()
# with sess.as_default() as sess:
#     print("Test example:\n")
#     y_true = K.constant(np.array(
#         [[1, 1, 1],
#          [1, 1, 0],
#          [0, 1, 0],
#          [1, 0, 1]]
#     ))
#     print("y_true:\n")
#     print(y_true.eval())
# 
#     w_p = np.array([0.25, 0.25, 0.5])
#     w_n = np.array([0.75, 0.75, 0.5])
#     print("\nw_p:\n")
#     print(w_p)
# 
#     print("\nw_n:\n")
#     print(w_n)
# 
#     y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
#     print("\ny_pred_1:\n")
#     print(y_pred_1.eval())
# 
#     y_pred_2 = K.constant(0.3*np.ones(y_true.shape))
#     print("\ny_pred_2:\n")
#     print(y_pred_2.eval())
# 
#     # test with a large epsilon in order to catch errors
#     L = get_weighted_loss(w_p, w_n, epsilon=1)
# 
#     print("\nIf we weighted them correctly, we expect the two losses to be the same.")
#     L1 = L(y_true, y_pred_1).eval()
#     L2 = L(y_true, y_pred_2).eval()
#     print(f"\nL(y_pred_1)= {L1:.4f}, L(y_pred_2)= {L2:.4f}")
#     print(f"Difference is L1 - L2 = {L1 - L2:.4f}")

# In[71]:


import tensorflow.keras.backend as K
K.tensorflow_backend.set_session(sess)

sess = K.get_session()
with sess.as_default() as sess:
    print("Test example:\n")
    y_true = K.constant(np.array(
        [[1, 1, 1],
         [1, 1, 0],
         [0, 1, 0],
         [1, 0, 1]]
    ))
    print("y_true:\n")
    print(y_true.eval())

    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    print("\nw_p:\n")
    print(w_p)

    print("\nw_n:\n")
    print(w_n)

    y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
    print("\ny_pred_1:\n")
    print(y_pred_1.eval())

    y_pred_2 = K.constant(0.3*np.ones(y_true.shape))
    print("\ny_pred_2:\n")
    print(y_pred_2.eval())

    # test with a large epsilon in order to catch errors
    L = get_weighted_loss(w_p, w_n, epsilon=1)

    print("\nIf we weighted them correctly, we expect the two losses to be the same.")
    L1 = L(y_true, y_pred_1).eval()
    L2 = L(y_true, y_pred_2).eval()
    print(f"\nL(y_pred_1)= {L1:.4f}, L(y_pred_2)= {L2:.4f}")
    print(f"Difference is L1 - L2 = {L1 - L2:.4f}")


# In[72]:


df.corr()


# In[73]:


corrMatrix = df.corr()
print (corrMatrix)


# In[74]:


sns.heatmap(corrMatrix, annot=True)
plt.show()


# # DenseNet121
# Next, we will use a pre-trained DenseNet121 model which we can load directly from Keras and then add two layers on top of it:
# A GlobalAveragePooling2D layer to get the average of the last convolution layers from DenseNet121.
# A Dense layer with sigmoid activation to get the prediction logits for each of our classes.
# We can set our custom loss function for the model by specifying the loss parameter in the compile() function.

# # create the base pre-trained model
# base_model = DenseNet121(weights= None, include_top=False)
# 
# x = base_model.output
# 
# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)
# 
# # and a logistic layer
# predictions = Dense(len(labels), activation="sigmoid")(x)
# 
# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

# # Checking out different types of models

# In[75]:


from torchvision import models
import torch

dir(models)


# In[66]:


alexnet = models.alexnet(pretrained=True)


# In[67]:


print(alexnet)


# In[68]:


from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


# In[69]:


# Import Pillow
from PIL import Image
img = Image.open("/Users/kellymclean/COVID_19_Lungs/images/1e64990d1b40c1758a2aaa9c7f7a85_jumbo.jpeg")
# summarize some details about the image
print(img.format)
print(img.mode)
print(img.size)
# show the image
#img.show()


# from matplotlib import image
# from matplotlib import pyplot
# data = image.imread("/Users/kellymclean/COVID_19_Lungs/images/1e64990d1b40c1758a2aaa9c7f7a85_jumbo.jpeg")
# print(data.dtype)
# print(data.shape)
# pyplot.imshow(data)
# pyplot.show()

# In[70]:


img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)


# In[71]:


alexnet.eval()


# In[72]:


#interference
out = alexnet(batch_t)
print(out.shape)


# In[73]:


with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


# In[74]:


_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

#print(labels[0], percentage[index[0]].item())


# In[75]:


print(labels[20], percentage[20].item())


# 1e64990d1b40c1758a2aaa9c7f7a85_jumbo.jpeg is Lymphocytic Interstitial Pneumonia

# In[76]:


# First, load the model
resnet = models.resnet101(pretrained=True)

# Second, put the network in eval mode
resnet.eval()

# Third, carry out model inference
out = resnet(batch_t)

# Forth, print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100


# In[77]:


print(labels[20], percentage[20].item())


# In[78]:


densenet = models.densenet121(pretrained=True)


# In[79]:


from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


# In[80]:


# Import Pillow
from PIL import Image
img = Image.open("/Users/kellymclean/COVID_19_Lungs/images/1e64990d1b40c1758a2aaa9c7f7a85_jumbo.jpeg")


# In[81]:


# create flipped versions of an image
from PIL import Image
from matplotlib import pyplot
# load image
#image = Image.open('/Users/kellymclean/COVID_19_Lungs/images/1e64990d1b40c1758a2aaa9c7f7a85_jumbo.jpeg"')
# horizontal flip
hoz_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
# vertical flip
ver_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
# plot all three images using matplotlib
pyplot.subplot(311)
pyplot.imshow(img)
pyplot.subplot(312)
pyplot.imshow(hoz_flip)
pyplot.subplot(313)
pyplot.imshow(ver_flip)
pyplot.show()


# In[82]:


img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)


# In[83]:


densenet.eval()


# In[84]:


#interference
out = densenet(batch_t)
print(out.shape)


# In[85]:


with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


# In[86]:


_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

#print(labels[0], percentage[index[0]].item())


# In[87]:


print(labels[20], percentage[20].item())


# In[88]:


#DENSENET 121 is much better than alexnet or Resnet


# In[89]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


# import tensorflow as tf
# 
# import pandas as pd
# import numpy as np
# import os
# import keras
# import random
# import cv2
# import math
# import seaborn as sns
# 
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# 
# import matplotlib.pyplot as plt
# 
# from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
# from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
# 
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.applications.densenet import preprocess_input
# 
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
# 
# from tensorflow.keras.models import Model
# 
# from tensorflow.keras.optimizers import Adam
# 
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# 
# import warnings
# warnings.filterwarnings("ignore")

# In[10]:


import tensorflow
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# #Add in metrics in order to get AUC and ROC
# 
# 
# METRICS = [
#       tensorflow.keras.metrics.TruePositives(name='tp'),
#       tensorflow.keras.metrics.FalsePositives(name='fp'),
#       tensorflow.keras.metrics.TrueNegatives(name='tn'),
#       tensorflow.keras.metrics.FalseNegatives(name='fn'), 
#       tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
#       tensorflow.keras.metrics.Precision(name='precision'),
#       tensorflow.keras.metrics.Recall(name='recall'),
#       tensorflow.keras.metrics.AUC(name='auc'),
# ]

# In[76]:


# create the base pre-trained model
base_model = DenseNet121(weights='imagenet', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))


# # create the base pre-trained model
# base_model = densenet121(weights=None, include_top=False, input_shape=(270, 480, 3))
# 
#     # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
#     # add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
#     # output layer
# predictions = Dense(session.training_dataset_info['number_of_labels'], activation='softmax')(x)
#     # model
# model = Model(inputs=base_model.input, outputs=predictions)
# 
# learning_rate = 0.001
# opt = keras.optimizers.adam(lr=learning_rate, decay=1e-5)
# 
# model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy', METRICS])

# my_callbacks = [
#     tensorflow.keras.callbacks.EarlyStopping(patience=2),
#     tensorflow.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
#     tensorflow.keras.callbacks.TensorBoard(log_dir='./logs'),
# ]
# model.fit(dataset, epochs=10, callbacks=my_callbacks)

# In[78]:


history = model.fit_generator(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 3)

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()


# # 5 Prediction and Evaluation
# Now that we have a model, let's evaluate it using our test set. We can conveniently use the predict_generator function to generate the predictions for the images in our test set.
# Note: The following cell can take about 4 minutes to run.

# In[80]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




