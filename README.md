# Project Haruki Murakami
This is a little text generation project about using transformers to produce Haruki Murakami style article. 
![Image](https://img.theculturetrip.com/1440x807/smart/wp-content/uploads/2011/08/screen-shot-2016-10-05-at-3-11-54-pm.png "Haruki Murakami")
*Image of Haruki Murakami by [article](https://theculturetrip.com/asia/japan/articles/japan-caught-between-cultures/) from [Hazel Rowland](https://theculturetrip.com/authors/hazel-rowland/)*.
## Introduction
Since the very beginning I started to learn data science and machine learning, text generation has been a attractive area to me. Just imagine that one day AI will be able to produce high quality articles or even literature novels like novelist in the future always make me feel excited. Up to the present, in the drawing area, AI products like [stable diffusion](https://stability.ai/blog/stable-diffusion-public-release) and [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F) have got incredible sucess, it is then inevitable to expect that one day Ai can be used to write literature novel with real plot as well. As a literature lover and a NLP enthusiast, I made this project earlier before as a first step to fulfill my dream. 

## Description of project
As shown in the file model.py, I basically used a conventional encoder part of the transformer and take a method called maximum likelihood estimation (MLE) to implement text generation task. Furthermore, the optuna is used as shown in file fine_tune.ipynb to find appropriate hyperparameters and optimization strategy. 

It is noted that unlike the familar encder setup like BERT having 12 layers, 12 multi-attentions, and 512 hidden dimensions, the encoder here have 10 layers, 8 multi-attentions, and 1024 hidden dimensions instead. The difference originated from the curiosity of the author about how is the final result quality if the original structure of model changes. In the long run, the training result does not look better obviously. The model captures little sementaic meaning of the sequences and the result losses coherence and have serious degeneration problem as expected. Also, from the loss picture shown in the loss picture directory implies that the model is not good enough so the generation peroforms poorly even we set 50 epochs.   

## Future improvement
Plural adjustments are considered and may be implemented in the future:
* **Improvement on hyperparameters:** In this project, I set the batch size to be 8, which is little bit too small actually. Bigger batch size has been proved to be better while training since model not only train faster but also has more information in single batch for model to make prediction. The hidden hidden may be set to be smaller so the word embedding space is denser and the relation between each words may be determined much accurately.
* **Quantity of data:** The cotpus and dictionary for model to train and predict are all from the words in the Murakami's excellent novel『色彩を持たない多崎つくると、彼の巡礼の年』. People are encouraged to read the novel and Murakami's other fascinating works.
