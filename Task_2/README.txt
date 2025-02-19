To create an image classification model, we used transfer learning. The model RasNet50 wich based on the imagenet weights, 
after tuning it showed higher accuracy rates(аccuracy increased from 57 to 93) than CNN from scratch.


A dataset with animal life was not found, so data for the NER model was artificially created in small quantities. 
This fact affected the results of the model. The Spacey library was used to train the NER model.
