import pandas as pd
import cv2
from torchvision import transforms
from tqdm import tqdm
import os

from utils.Utils import extractValue


class Dataset:
    TRAIN_PATH = 'data/train/'
    TEST_PATH = 'data/test/'

    def __init__(self, size=224):
        self.train = []
        self.test = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((227, 227), antialias=True),
        ])

        df = pd.read_csv('data/y_train.csv')

        #initialize train data
        for file in tqdm(os.listdir(self.TRAIN_PATH), desc='Extracting and transforming values from train data'):
            img = cv2.imread(self.TRAIN_PATH+file)
            if img is not None:
                img = self.transform(img)
                filename = os.path.splitext(file)[0]
                label = extractValue(df, filename, 'species', True)
                self.train.append((img, label))

        df = pd.read_csv("data/y_test.csv")
        #initialize test data
        for file in tqdm(os.listdir(self.TEST_PATH), desc='Extracting and transforming values from test data'):
            img = cv2.imread(self.TEST_PATH+file)
            if img is not None:
                img = self.transform(img)
                filename = os.path.splitext(file)[0]
                label = extractValue(df, filename, 'species', True)
                self.test.append((img, label))

        print('Data loaded succesfully!')


    
        