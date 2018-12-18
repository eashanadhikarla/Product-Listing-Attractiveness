## Is Your Product Listing Attractive?

The quality of product listing is crucial for improving search relevance and gaining customer attention.

# INTRODUCTION
To build a model to automatically predict the quality of set of product listing. I used a keras tensorflow model to learn the train data from it’s behaviour with the help of the following features;

● name (Name of the product)
● description (Description about the product)
● lvl1, lvl2 and lvl3 (Are the branches of categorization of each product)
● price (Price of the products)
● type (International or a Local product)

# PROBLEM STATEMENT
“​On E-commerce sites, the quality of product listing is crucial for improving search relevance and gaining customer attention. In this competition, you are provided a set of product names, description, and attributes, as well as the quality score of their listing as rated by real customers.​"

STAGE 1 - DATA CREATION / PREPARATION
1. Methods, I used for data cleaning includes :
Tokenizer package from keras’s processing text package. Where tokenizer has an attribute “filter” which filters all the special characters in the numpy array. This is why we will also have to first convert the training data from pandas dataframe to numpy array or pandas Series. Keras text processing has a buffer which takes block wise numpy input known as tokens and filters it. Other things I tried for filtering includes :
Replace technique with regex expression for removing special characters. This glimpse of code is given below;
This method works perfectly fine with both numpy arrays and pandas dataframes.
    
df_test.replace(regex​=​{​'<.*?>|&nbsp|\W'​: ​' '​}, inplace​=​True​)
 
2. For feature engineering, I started with turning the characters of each word into lower case strings for removing the ambiguity of considering the “Laptop” and “laptop” as different things.
3. I also used feature scaling technique in my file “feature_engineering.ipynb”. Where using min() max() function on data feature “price” helped me a lot to increase my accuracy. I would rate it as an important feature as the accuracy change from drastic form log loss 1.10024 to log loss ​0.63879​.
