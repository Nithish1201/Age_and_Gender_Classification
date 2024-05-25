import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
import cv2

# Python program to store list to file using pickle module
import pickle

cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

imdb_path = r'E:\age and gender prediction\project\data\imdb_crop\imdb.mat'
wiki_path = r'E:\age and gender prediction\project\data\wiki_crop\wiki.mat'

imdb_mat = imdb_path
wiki_mat = wiki_path

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('imdb_crop/' + path[0])

for path in wiki_full_path:
    wiki_path.append('wiki_crop/' + path[0])

imdb_genders = []
wiki_genders = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
    else:
        imdb_genders.append('female')

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')

imdb_dob = []
wiki_dob = []

for file in imdb_path:
    temp = file.split('_')[3]
    temp = temp.split('-')
    if len(temp[1]) == 1:
        temp[1] = '0' + temp[1]
    if len(temp[2]) == 1:
        temp[2] = '0' + temp[2]

    if temp[1] == '00':
        temp[1] = '01'
    if temp[2] == '00':
        temp[2] = '01'
    
    imdb_dob.append('-'.join(temp))

for file in wiki_path:
    wiki_dob.append(file.split('_')[2])


imdb_age = []
wiki_age = []

for i in range(len(imdb_dob)):
    try:
        d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    imdb_age.append(diff)

for i in range(len(wiki_dob)):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    wiki_age.append(diff)

final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T
final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

meta = pd.concat((final_imdb_df, final_wiki_df))

meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']

meta = meta.drop(['face_score1', 'face_score2'], axis=1)

meta = meta.sample(frac=1)

meta.to_csv('meta.csv', index=False)

path = r'E:\age and gender prediction\project\meta.csv'
data = pd.read_csv(path)

over_age_lst = data[data['age'] > 90].index.tolist()
under_age_lst = data[data['age'] < 4].index.tolist()
data.drop(over_age_lst, inplace= True, axis = 0)
data.drop(under_age_lst, inplace= True, axis = 0)
data.reset_index(drop=True, inplace=True)

bins = [0, 13, 34, 56, np.inf]
names = ['Kid', 'Youth', 'Adult', 'Old']

data['AgeRange'] = pd.cut(data['age'], bins, labels=names)

data.to_csv('cleaned_IMDB_WIKI.csv',index = False)

def isgray(imgpath):
    img = cv2.imread(imgpath)
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

gray_index_lst = []
for index,i in zip(data.index,data['path']):
    if isgray('/age and gender prediction/project/data/'+ i) == True:
        gray_index_lst.append(index)

# write list to binary file
def write_list(a_list):
    # store list in binary file so 'wb' mode
    with open('IMDB_WIKI_gray_img_list', 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('IMDB_WIKI_gray_img_list', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

write_list(gray_index_lst)
gray_img_index = read_list()

dataset = data.copy()
dataset.drop(gray_index_lst, axis = 0, inplace =True)
dataset.reset_index(drop=True, inplace =True)

dataset.to_csv('cleaned_IMDB_WIKI_non_gray.csv',index = False)

data1 = dataset.sample(frac=.01)

data1.to_csv('sample_IMDB_WIKI_non_gray.csv', index = False)