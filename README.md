# (WIP) Size recommender system
We want to build a model able to recommend the best fitting size for a given pair of user and item (brand + model + gender). We approach it as a collaborative filtering problem but, in this case, havin multiple sizes for each item complicates thinsg a little bit


## Available features

### User features

- `user_id`
- foot shape (categorical)
- gender
- number of years:
	- bouldering
	- sport climbing
	- trad climbing
- street shoe size (US or EURO)

### Item features
- brand
- model
- gender
- size (US or EURO)

### Target variable
Rating (between 1 and 5) for a given pair of user and sku (brand + model + gender + size)

## Feature engineering
It seems pretty natural to use embedding for `user_id`. Hopefully we can learn a useful low dimensional representation of the user that, when *combined* with a sku embedding can help oredict the rating. The intuition behind this is that if we had access to the dimensions of the user feet and the shoes, as well as some detailed information about the user fit preferences we would be able to derive a model to predict the rating. Here we hope for the model to learn these relationships.


## Model architecture
### Output layer

Given and pair of user and sku, We want to predict the rating. To do so we can either treat this as a regression problem, predicting the value of the rating or as a multi-class classfication problem, predicting the probability for each class. It is not immediately obvious to me which method is better. The classifier offers the advantage that will have a prediction for each rating/class, possibly allowing us to estimate the confidence in the prediction while the regression model takes into account the ordinal relationship between the rating, which could make the learning task easier.








## Size augmentation


The data we have present a few difficulties:
- if computed naively, the user/sku (item + size) matrix is very sparse (<1%)
- we do not have an exhaustive list of available sizes for each shoe, complicating the task 
- we have a mix of US and EURO sizes but no mapping between the sizes
- the US/EURO mapping is gender depedent


## Improvements
- **Allow gender to be part of recommendation**: shoe gender is both about aspect and foot shape. Some users who identify as a given gender might find, for some specific item, a better fit from a shoe with another gender than theirs.
-
