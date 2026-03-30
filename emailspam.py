import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
nltk.download('stopwords')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("Before reading CSV")
data = pd.read_csv("spam_ham_dataset.csv")
print("CSV loaded successfully")
print(data.head())
print(data.shape)

sns.countplot(x='label', data=data)
plt.show()

ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)
sns.countplot(x='label', data=balanced_data)
plt.title("Balanced Distribution of Spam and Ham Emails")
plt.xticks(ticks=[0, 1], labels=['Ham (Not Spam)', 'Spam'])
plt.show()