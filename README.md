# face-recognition
## The competition's baseline and data can be download from the url.
链接：https://pan.baidu.com/s/1jYWaFjhGEC1zxcWw6SVl5Q 
提取码：gu09 
## Face recognition  with resnet18，resnet50 and resnet 152.  
The dataset is very small, so it is really easy to overfitting   
I use some tricks to reduce the effect of overfitting, such as dropout, regularization and model intergration.  
Of course, there are some preprocessing operations for images.  
I have tried data augumentation, tta and grey image, but they make no sense.  
In fact, It's not strictly a data set of faces,background is a big part of it and there are many images with more than one person ,which make face recognition difficult. Although I tried to cut out faces first, but because of other people's interference, the result became worse.
