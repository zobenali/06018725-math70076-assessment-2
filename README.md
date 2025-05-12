#  Painting Clustering : Predicting A painting movement using Deep Learning

Repository for my project in the context of me classes of IMA206 at Télécom Paris and MATH70076-Data Science at Imperial College London

This project is to me the first milestone of trying to make art more accessible to a large public. Indeed, people coming from different social cast does not have the same cultural background or educational background and some does not have access through their surroundings or environments to art sensibility. But cultural background does influence social climbing. Therefore, me and my friends in the context of IMA206 Project at Télécom Paris tried to develop an algorithm able to detect a painting's movement. We came up with pretty strong results and some great prediction.

As I really enjoyed working on it, I revisited this project for my MATH70076- Data Science class at Imperial College trying to make it usable for everyone and reproducible. The project came from a small Jupyter Notebook and turned into this repository. 

Feel free to use it.

_This project uses Python_

## Prerequisites
You can set up the project using : 

```sh
   git clone https://github.com/zobenali/painting_clustering
```

Make sure you also have all requirements including torchvision. You can use the following to run the project : 

```sh
   pip install -r requirements.txt
```
## Project Structure

The data folder cannot contain the data I used as they are too voluminous. Furtehrmore, they are part of a private collection so I won't make them public. However, there are a lot of paintings databases that you can use for this project as the code conatains all pre-processing functions. I linked some in data folder so you can make use of the project too.

The src folder contains all the code required to run the full project.

 - data_processing.py  : contains function to prepare the data from folders to images -- Do change the function create_data_frame if your datas does not match the structure I used

 - model.py : creates the base model for the training. 
 - train.py : trains and saves the model in the output directory, the __main__ also plots the loss 
 - eval.py : tests the data
 - gmm_analysis.py : performs the tsne and gmms plots for clustering
 - movement.py : takes a picture or batch of picture and predicts their most likely to belong movement

 The outputs folder contains figures and model I obtained with this code.

 The report contains my Reflective_Summary in the context of my Imperial Course

## Usage 
**To write**



## Contributions
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


