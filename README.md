# Tensorflow-Resnet-Image-Classification
这是之前我和我的组员参与Tiny-mind 手写汉字识别大赛的一个项目， 主要实现了用Resnet或VGG16作为Backbone来训练一个分类任务以及基于opencv的一些图片数据集增强，最终我们组的成绩为TOP-5准确率99.19， 排名5/700。这个项目也可以用于其他的图片分类任务，整个网络框架是基于Tensorflow实现的
## 数据集格式
训练数据集和测试数据集分别存放在data/train和data/test2文件夹下，数据集格式：
```
├── data
      ├── train
      │   ├── 夏
      │       ├── 001.jpg
      │       ├── 002.jpg
      |       ├── 003.jpg
      │   ├── 武
      │       ├── 004.jpg
      │       ├── 005.jpg
      |       ├── 006.jpg
      │   ├── 陈
      │       ├── 007.jpg
      │       ├── 008.jpg
      │       ├── 009.jpg
      ......
      ├── test2
```
对于每一个类别，其文件夹名称就是类别名称<br>
## 数据增强
```
cd $PATH_ROOT
python data_update.py
```
进行训练数据增强的操作，主要包括图像的旋转，仿射变换，伽马变化，随机高斯噪声，运行完成后data文件夹下的图片目录会保存相应的增强数据图片。
## 训练
```
cd $PATH_ROOT
python train.py
```
进行网络的训练，在训练之前可以修改config.py文件对网络相应的配置进行修改，如训练步数，L2正则化系数，网络结构（Resnet50,Resnet101,Resnet152）进行修改。<br>
summay文件夹下存放的是tensorboard的缓存
## 测试
```
cd $PATH_ROOT
python test.py
```
进行训练好的模型测试，在测试之前可以修改output_2文件夹下，修改checkpoint文件中model_checkpoint_path的值来指定测试不同步数下的模型。最终$PATH_ROOT下会生成一个data.txt文件和一个res_test.csv文件，data.txt文件存放的是每一张图片top5 类别的score， res_test.csv是赛事要求提交的csv文件。<br>
关于测试可视化：<br>
取消注释test.py文件中
```
# print(name_label) //python
# cv.imshow('sprint（image_datahow', test_image_single_1)
# cv.waitKey(0)
```
单张显示图片并打印top5的分数
