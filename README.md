# (WIP) Size recommender system
We want to build a model able to recommend the best fitting size for a given pair of user and item (brand + model + gender). We approach it as a collaborative filtering problem but, in this case, havin multiple sizes for each item complicates thinsg a little bit


## Size augmentation


The data we have present a few difficulties:
- if computed naively, the user/sku (item + size) matrix is very sparse (<1%)
- we do not have an exhaustive list of available sizes for each shoe, complicating the task 
- we have a mix of US and EURO sizes but no mapping between the sizes
- the US/EURO mapping is gender depedent


## Improvements
- **Allow gender to be part of recommendation**: shoe gender is both about aspect and foot shape. Some users who identify as a given gender might find, for some specific item, a better fit from a shoe with another gender than theirs.
-
