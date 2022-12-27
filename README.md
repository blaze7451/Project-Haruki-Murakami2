# Project Haruki Murakami
This is a little text generation project about using transformers to produce Haruki Murakami style article. 
![Image](https://img.theculturetrip.com/1440x807/smart/wp-content/uploads/2011/08/screen-shot-2016-10-05-at-3-11-54-pm.png "Workflow of a generic IR-based QA system")
*Image of Haruki Murakami by [article](https://theculturetrip.com/asia/japan/articles/japan-caught-between-cultures/) from [Hazel Rowland](https://theculturetrip.com/authors/hazel-rowland/)*.
## Introduction
Since the very beginning I started to learn data science and machine learning, text generation has been a attractive area to me. Just imagine that one day AI will be able to produce high quality articles or even literature novels like novelist in the future always make me feel excited. Up to the present, in the drawing area, AI products like [stable diffusion](https://stability.ai/blog/stable-diffusion-public-release) and [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F) have got incredible sucess, it is then inevitable to expect that one day Ai can be used to write literature novel with real plot as well. As a literature lover and a NLP enthusiast, I made this project as a first step to fulfill my dream. 

## Description of project
As shown in the file model.py, I basically used a conventional encoder part of the transformer and take a method called maximum likelihood estimation (MLE) to implement text generation task. It is noted that unlike the familar encder setup like BERT having 12 layers, 12 multi-attentions, and 512 hidden dimensions, the encoder here have 10 layers, 8 multi-attentions, and 1024 hidden dimensions instead. The difference is  

## Future improvement
