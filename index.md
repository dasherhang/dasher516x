---
layout: default
---

Text can be **bold**, _italic_, or ~~strikethrough~~.


### Interesting question 

With the increasing world population, the amount of food requires to feed the population inevitably increases too. This means that livestock cultivation needs to keep up with the demand, and increasing farming efficiency is the priority in the agricultural field. There's a need for new technology in the field to help increase the production efficiency of livestock. Many livestock farmers rely on traditional methods like manual auditing by workers and tracking the number of livestock they have. This poses many challenges as humans are prone to making errors, with the number of livestock in a farm reaching thousand of hundreds, the increase the complexity of livestock auditing. The scientific goal is to achieve the auditing of livestock numbers with a non-invasive, cost-effective, easy implementation, and maintainable method in auditing livestock numbers. This project's scope is to train a computer vision-based deep learning model for the classification of livestock in a farm setting and auditing the number. As with most machine learning data, having a large amount of data might increase the model's performance.

### Data description

The data used for this project is video data of livestock moving through an area from a top-down view and capturing a bird's eye view of the livestock. The data were obtained from a farm. The data are uploaded and added To a Google Drive to allow easy access by anyone needing it, as anyone can be added to the shared folder and download the clip for usage. The issue with the data is that human is present in the video, and to prevent any future issue, the video data are viewed, and any clip with human faces are removed from the dataset. The video data needs to be transformed into individual pictures to train the computer vision-based deep learning model. It's usually transformed into the number of frame rates it's filmed in. For example, a 5s video filmed at 30FPS will yield 150 pictures. These pictures need to be annotated, meaning a square box called a "bounding box" captures all the livestock within the box. Data annotation can be done by any software used for deep learning training, as it should contain the original picture and a file of all the livestock's bonding boxes within the specific picture. The one used for this project is Roboflow, which provides an API key that can directly retrieve the training data into any environment. This data will not change over time but can be added to increase the training size. Roboflow also allows users to set a different version of training data to include a different split of the train, valid, and test data. 


Data Transformation: The picture data for training the model are transformed into a different orientation of the original data to allow the model more variation in training, making it better at predicting objects it has never seen. 


Class concept: The data preparation used concepts learned like cloud computation, like Google Collaboratory. It's similar to the Jupyter Notebook but with the added capability to easily share the notebook with anyone with a Google account and contribute to it. On top of that, google drive and being directly mounted to the notebook allow easy access to any data needed for the project, and everyone with access to the shared google drive can access data and use it, making the research highly reproducible and allowing collaboration within a research team. Any output the notebook produces can also be saved into google drive directly. Another benefit of cloud computing using Google Colab is that it can be accessed directly on the browser with a Google account set up, saving the hassle of setting up the programming environment on a local computer and having to download and install libraries and packages, as google collab have access to the latest version to most libraries and packages for Python. Google Colab also can use virtual GPU and CPU to increase the computation power when dealing with a large number of dataset and training mode that requires large computation power, reducing the need to own a powerful computer. 

### Model the data

The deep learning model used for this project is a computer vision-based model called YOLOv5; it's a one-stage object detection model that detects and finds the location of the objects in a single pass of the input picture. It's essentially a Convolutional Network Neural base model that has the backbone to collect image features, a neck to create different layers of feature maps and group the information of which feature the input has, and a head uses this features collection to predict the object and the location it at and also the confidence score of how likely the predicted object is the actual object. To validate the model, we look at the Precision, Recall, and Mean Average precision mAP. Confusion Metric is a better metric to evaluate the model than most text-based models. 

Precision = TP/(TP+FP), meaning if the model predicted 10 livestock in the area, how many of the 10 predictions are livestock? If the actual count is 10 out of 10, the Precision will be 1. If it's 5/10, the precision will be 0.5, the higher, the better. 

Recall = TP/(TP+FN), meaning if the model predicted 10 livestock, but there are 20 livestock in the area, 10/20, the precision will be 0.5; the model missed half of the livestock in the area. If the model didn't fail to predict 20 of the livestock in the area, meaning False Negative is 0, 20/20 showing that the recall will be 1, same here the higher the recall, the better the model performs. 

A Mean Average Precision would be a more appropriate model to evaluate the model as. A  high mAP means the model has a low false negative(FN) and false positive (FP) rate. 

Once the model has been trained, a separate set of videos are used to run inference. The output will be a file with the location of the detected object, and pictures with annotations on each detected object for visualization that the model is predicting correctly 


Class concept: 
1)It still follows the train, valid, and test split of data similar to training most of the models we learned in class.
2) False Positive is known as a Type 1 error where the model thinks there's an object in the area, but there's nothing. A False Negative is known as a Type 2 error where the model didn't detect the existence of an object. 

###Output Data Preparation for Auditing

After detection is down, a second algorithm is used to associate all the livestock's locations in each frame to track the position and get the final number of livestock in the video. To prepare the data, each file's name is modified and added to the dictionary so that it can be called and used throughout the auditing phase. Each file's contents are put into a Pandas data frame so the location of each file can be easily retrieved. 

Class Concept: The use of the text wrangling technique help edit the files' name, and also, Pandas Dataframe learned in class highly aids the ability to retrieve data. 

### Audit Output and comparison
An Auditing script provided a final number of livestock in the video data. The final number is recorded and compared to the ground truth in a table. 

A final data with the ground truth number and number from the script are used to plot in a scatter plot with a regression line, and an OLS regression is run to see how accurate each algorithm number is relative to the ground truth. For the plot, if the counts are mostly accurate, all the dots should be close to the line. The R-squared provided by the OLS gives a good indication of the regression line fitting the data. A regression benefits from examining the correlation of Ground Truth and Algorithm number as it can show the researcher a trend. If the sample size increase, will the algorithm's accuracy drop? Using a regression model opens up an opportunity to examine what other factors affect the number's accuracy, like the light source, background, livestock body size, color, etc.. All these could be a variable to help researchers identify which factors have the most effect in contributing to an inaccurate number when these data became available.

Class Concept: Understanding the correlation between variables and Utilising Linear regression to evaluate how accurate the number is.

### Communciate and visualize the results

The correlation plot of ground truth shows the effectiveness of utilizing a non-invasive method to audit the livestock number, which allowed further study into the area. I learned that using a model regression could benefit future research to identify what factors affect an inaccurate number when data are available 



```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
