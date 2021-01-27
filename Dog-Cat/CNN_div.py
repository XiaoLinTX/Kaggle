'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-11 20:01:10
LastEditors: ZhangHongYu
LastEditTime: 2021-01-27 11:16:21
'''
import shutil
import os


#统计图片个数
train = '/media/lonelyprince7/3cb87c41-9244-4173-b0bf-d889261210f3/kaggle/dog-cat/origin/train/'

dogs = [train+i for i in os.listdir(train) if 'dog' in i]

cats = [train+i for i in os.listdir(train) if 'cat' in i]

print(len(dogs), len(cats))


#图片分类


def createDir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("创建文件夹失败")
            exit(1)


path = "/media/lonelyprince7/mydisk/CV-dataset/dog-cat/"

createDir(path+"train/dogs")
createDir(path+"train/cats")
createDir(path+"test/dogs")
createDir(path+"test/cats")

for dog, cat in list(zip(dogs, cats))[:1000]:
    shutil.copyfile(dog, path+"train/dogs/"+os.path.basename(dog))
    print(os.path.basename(dog)+"操作成功")
    shutil.copyfile(cat, path + "train/cats/" + os.path.basename(cat))
    print(os.path.basename(cat) + "操作成功")
for dog, cat in list(zip(dogs, cats))[1000:1500]:
    shutil.copyfile(dog, path + "test/dogs/" + os.path.basename(dog))
    print(os.path.basename(dog) + "操作成功")
    shutil.copyfile(cat, path + "test/cats/" + os.path.basename(cat))
    print(os.path.basename(cat) + "操作成功")
