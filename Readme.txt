Team Members: 
1. Prajwal Srinivas Srinath
2. Rajath Vanakudare
3. Shreyas Ramkumar Karthik

Project Description:
Our project aims to comprehend the content of a given image and caption the image based on the contents.
This is a combination of an object detection task and NLP task, ie, framing a caption (comprehending the image in words) based on the detected objects in the image. We have analyzed 3 pre-trained models such as ResNet-50, InceptionV3 ,and VGG-16 object recognition models, we remove the last layer and feed the embeddings to an LSTM which can then be further trained to predict captions based on the
detected objects in the image. We compare ResNet50, VGG16, and InceptionV3 for object recognition while keeping the LSTM component constant. We intend to compare these pre-trained networks using the
BLEU (BiLingual Evaluation Understudy) evaluation metric and utilize the flickr8k dataset for
completing the task.

Extension:
* Built a web application(GUI based application) using flask to upload an image and get captions from all the 3 models(Can be seen in demo).
* Implemented InceptionV3 model to compare the results(This was not given in the initial project proposal but we have implemented this to analyze the results)

URL for presentation and Demo:
