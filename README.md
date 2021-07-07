# Sign_Recognition_Sentiment_Analysis

# Background and Introduction
With the advent of Deep learning developing Recommender systems is highly in demand as it makes the life easy in various fields such as Medical Science, Education, Automobiles, Aviation etc. We came up with a project from one of our clients to build Sign Recognition system for Dumb people as requested by one of the prominent Educational institutes. Understanding the feelings, emotions or messages of dumb people is necessary as it becomes important on the part of observer that the sentiments are not misinterpreted.

In this project we came up with an idea of building a system that is trained on the frames or images as dataset. Each of these frames or images consist of signs captured in videos by the participant. We apply the trained CNN model on the received image and try to define the letter conveyed by the sign of the participant. After sign extraction the sentences are framed using the signs. Now the sentences convey the message that the participant wants to share with the other person.

Using the sentences received from the user we try to decode the sentiments of the speaker by applying various Natural Language Technique process. We have build a model using the Twitter Tweets Sentiment dataset that classifies the tweets as Positive and Negative emotions. Once the model is ready we apply it on the incoming sentence from the participant. The model then will classify the sentence from the user. Thus an end-to-end system is designed that helps to understand the impaired people in a more corrective and convenient way.



#	Software Architecture
![image](https://user-images.githubusercontent.com/42905724/124798677-43fec280-df71-11eb-90b6-c5909bfbae85.png)





#	Sign Language Model Building using CNN
![image](https://user-images.githubusercontent.com/42905724/124798757-5ed13700-df71-11eb-9002-dfa558c8cb68.png)

American Sign Language is the representation of 26 alphabets of English using various hand gestures. Initially we obtain the raw images divided into train and test set. Each of the images are available in the raw form and we need to filter the image before sending them out to the CNN model for training. Below are the steps performed as part of image Preprocessing.





# Image Greyscale Normalization
Firstly we convert the image in the binary format as the CNN model performs better with 0 and 1 as inputs. This process is known as normalization.Next we reduce the pixels of image frame to 28*28 size and then apply the greyscaling on the images as a part of filtering.
![image](https://user-images.githubusercontent.com/42905724/124799190-e74fd780-df71-11eb-8d20-4cd141213f24.png)







# Image Augmentation
In order to avoid overfitting problem, we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations.

Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust mode.






# Creating CNN model
A Convolutional Neural Network has been constructed by passing the input images through various layers of maxpooling, creating conv2D layers and dropping out neurons from the network after each iteration. Also the end layer of CNN model is flattened and passed through softmax compiler to generate 24 outputs representing English Alphabets except ‘j’ and ‘z’.We then compile the model using the loss function as categorical crossentropy and optimization function as Adam.





# Model Building
We then split the dataset into training and testing pixel images. The dataset around 70% is allowed as training data and around 30% is set aside as testing dataset. We give the image inputs to the datagen that defines the data augmentation of the images. Also, batch size and epoachs to run the entire dataset is given as parameter in the model.fit function. 





# Output Results of accuracy and loss
The model output accuracy and loss is calculated based on the output of training and testing dataset. In the below images the accuracy and loss behaviour of training and testing dataset is displayed. It can be observed that the model shows more than 99% accuracy in both train and test dataset.

![image](https://user-images.githubusercontent.com/42905724/124799562-50374f80-df72-11eb-8e95-050bbfd00872.png)








#  Building Sentimental Analysis Model
The Sentiment Analysis model was build using the Twitter disaster Dataset which classifies the news as positive or negative based on the headlines. Initially we have a dataset with the headlines and the respective sentiment associated with it in a dataset. The twitter dataset has thousands of tweets from the user which is taken as input for building the Sentiment Analysis Model.

![image](https://user-images.githubusercontent.com/42905724/124799622-62b18900-df72-11eb-8e65-93718ba30c9a.png)








# Text Cleaning and Preprocessing
The raw data that is obtained in the dataset consists of so many punctuations , unwanted url header , hashtags which hardly contribute to determining the sentiment of a given sentence. So we need to apply the cleaning process on these sentences and remove the above unwanted characters which have null weightage as far as output is considered. After cleaning the sentences we store them in a corpus to which is a list of all sentences in the given dataset.








# One Hot Encoding
As a part of one-hot encoding we try to convert the words into numbers as it is easier for the machine to learn based on numbers rather than words. Here we  set the top n number of highest occurring words  in the entire corpus . Based on the occurrence number each of the words in a sentence are assigned a number associated with them. 






# Creation of LSTM model
LSTM model is a type of Recurrent Neural Network. When the output is time based and dependent on the previous sequenced output then it gives rise to LSTM. It latches onto the memory from the previous output and contributes to the 2nd  input in order to generate the output for the same. The tweets along with parameters such as size of vocabulary , and no of input neurons are passed to the LSTM layers and layers of dropout which generates the sigmoid ouput consists of two values 0 and 1 which represent the positive and negative sentence.







# Split the data and run the model on Training set
We need to next split the given dataset into train and test data and later run the model by setting the epochs and batch size.






# Model Accuracy
We obtained a very high rate of accuracy with the trainng set but the test set large variation in terms of accuracy. We need to work on  improving the accuracy of unknown tweets. The accuracy of test dataset is around 74.5% which is quite a large variation compared to train dataset that has more than 95% accuracy.
![image](https://user-images.githubusercontent.com/42905724/124799962-bb812180-df72-11eb-838e-36aaa20c3f2a.png)








# Combining both the models together
Our next step is to combine both the Sign recognition and sentiment analysis model to extract the sentence using the signs and try to predict the mood or emotion of the person. So basically we try to give the input of sentence in an excel sheet that represents the letters combined to form a sentence. The words are separated in a sentence by assigning a particular symbol to space value that defines the separation. Below is the output label which represents the alphabets in the images w.r.t any particular label.

![image](https://user-images.githubusercontent.com/42905724/124800007-c8057a00-df72-11eb-9da1-7c4f556cdca9.png)

Let’s take an example to better understand the formation of sentences from a given excel containing details of the pixels. Consider the below excel file as an input where each row represents either a letter or a space separating two words.Remember the label 23 represents space in the below snippet. Notice that the below image shows only 19 pixels however the image has 784 pixels which represents 28X28 image.
![image](https://user-images.githubusercontent.com/42905724/124800042-d2277880-df72-11eb-9721-9bce3d2031df.png)

We take the input file and convert into Dataframe using the pandas. Next the dataframe is converted into values and its shape is resized.
![image](https://user-images.githubusercontent.com/42905724/124800147-eff4dd80-df72-11eb-95e9-8bd4af3c3db2.png)

The model which is saved during the Sign Recognition designing is applied on the array of resized values to generate a prediction. The prediction is a single array representing the labels as values from 1 to 24 except value 9 representing ‘j’ as it has been ignored for our project purpose.
![image](https://user-images.githubusercontent.com/42905724/124800173-f8e5af00-df72-11eb-9b4a-a074e8430c66.png)


Next we loop through the output array named prediction and based on the for loop concat the letter associated with the label into sentence.
![image](https://user-images.githubusercontent.com/42905724/124800206-000cbd00-df73-11eb-9acf-ac1049f6c429.png)
![image](https://user-images.githubusercontent.com/42905724/124800217-0438da80-df73-11eb-85d2-09fd48ee4ef1.png)

Once the sentence is extracted we then call the Sentimental model. After applying all the word cleaning process and one hot encoding process on the sentence we pass it to the Sentimental Model. The model generates output as either 1 or 0 representing positive or negative emotion. 
![image](https://user-images.githubusercontent.com/42905724/124800249-0f8c0600-df73-11eb-8b5e-c4aa012b94c0.png)
![image](https://user-images.githubusercontent.com/42905724/124800264-1286f680-df73-11eb-938c-9134b00a49e5.png)











# Future Work
We need to improve the accuracy of for sentimental model of test dataset. Also, the ability to extract signs from live video frames must be implemented to help the product be more useful in live applications





# References


1.	Autonomous Multiple Gesture Recognition System for Disabled People https://www.researchgate.net/publication/272092040_Autonomous_Multiple_Gesture_Recognition_System_for_Disabled_People


2.	Assistive system for physically disabled people using gesture recognition https://www.researchgate.net/publication/321413153_Assistive_system_for_physically_disabled_people_using_gesture_recognition

3.	Assistive system for physically disabled people using gesture recognition https://ieeexplore.ieee.org/document/8124506

4.	How to perform Sentiment Analysis in Python 3 using the Natural Language toolkit(NLTK). http://www.iosrjournals.org/iosr-jce/papers/Vol18-issue4/Version-3/F1804033440.pdf 

5.	Automatic Number Plate Recognition System (ANPR): A Survey
https://www.researchgate.net/publication/236888959_Automatic_Number_Plate_Recognition_System_ANPR_A_Survey
     
6.	COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning  https://www.kaggle.com/madz2000/cnn-using-keras-100-accuracy?select=amer_sign2.png


7.	How do Convolutional Layers Work in Deep Learning Nerual Networks https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
     
8.	Dataset used : Sign Language MNIST
 https://www.kaggle.com/datamunge/sign-language-mnist

9.	Disaster Tweet Sentiment Analysis using LSTM
https://www.kaggle.com/charanrajshetty/disaster-tweet-sentiment-analysis-lstm#Split-the-independent-and-dependent-features


10.	 Sign-Language-Interpreter-using-Deep-Learning   
   https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning


