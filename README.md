# Sign_Recognition_Sentiment_Analysis

# Background and Introduction
With the advent of Deep learning developing Recommender systems is highly in demand as it makes the life easy in various fields such as Medical Science, Education, Automobiles, Aviation etc. We came up with a project from one of our clients to build Sign Recognition system for Dumb people as requested by one of the prominent Educational institutes. Understanding the feelings, emotions or messages of dumb people is necessary as it becomes important on the part of observer that the sentiments are not misinterpreted.

In this project we came up with an idea of building a system that is trained on the frames or images as dataset. Each of these frames or images consist of signs captured in videos by the participant. We apply the trained CNN model on the received image and try to define the letter conveyed by the sign of the participant. After sign extraction the sentences are framed using the signs. Now the sentences convey the message that the participant wants to share with the other person.

Using the sentences received from the user we try to decode the sentiments of the speaker by applying various Natural Language Technique process. We have build a model using the Twitter Tweets Sentiment dataset that classifies the tweets as Positive and Negative emotions. Once the model is ready we apply it on the incoming sentence from the participant. The model then will classify the sentence from the user. Thus an end-to-end system is designed that helps to understand the impaired people in a more corrective and convenient way.

#	Software Architecture
![image](https://user-images.githubusercontent.com/42905724/124798677-43fec280-df71-11eb-90b6-c5909bfbae85.png)
