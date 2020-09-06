```python

```

# Amazon and Flipkart Books - Price and Review Analysis

Amazon and Flipkart are the two leading E-commerece giants of India. Flipkart started its business in the year 2007 by selling books whereas amazon took its first steps in the indian market in the year 2012 by launching junglee.com(price comparision website) and later amazon's own online store, which also started by selling books mostly.

Flipkart did get an early headstart but the scenario is completely different now. A price analysis of some books being sold on both the platforms has been performed below to understand the competitive pricing done by the companies to maximise sales.

![](Images/potter.jpg)
![](Images/output_83_0.png)

## Part 1 - Price comparision

A kaggle dataset from the year 2018 has been used for the analysis which cotains nearly 700 unique records of randomly selcted books. ISBN or the International Standard Book Number is a numeric commercial book identifier which is intended to be unique.An ISBN is assigned to each separate edition and variation of a publication.
This uniquness of the ISBN number has helped joining and comparing books on both the platforms.

Who doesn't like saving a few bucks while shopping online ? We either have a favourite online store where we prefer landing directly or we scroll on google and then land on the website offering the cheapest price. In the following analysis I will be finding out how much would a bookworm be saving if he or she buys the entire dataset in a year or two. 


```python
#Importing the important libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
```


```python
amazon = pd.read_csv('amazon.csv')
flipkart = pd.read_csv('flipkart.csv')
```


```python
amazon.drop_duplicates(inplace=True)
amazon.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amazon_title</th>
      <th>amazon_author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_isbn-10</th>
      <th>amazon_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tell Me your Dreams</td>
      <td>by Sidney Sheldon</td>
      <td>4.4</td>
      <td>160.0</td>
      <td>8172234902</td>
      <td>209</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Boy in the Striped Pyjamas (Definitions)</td>
      <td>by John Boyne</td>
      <td>4.6</td>
      <td>134.0</td>
      <td>1862305277</td>
      <td>350</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Romancing the Balance Sheet: For Anyone Who Ow...</td>
      <td>by Anil Lamba</td>
      <td>4.5</td>
      <td>156.0</td>
      <td>9350294311</td>
      <td>477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mossad</td>
      <td>by Michael Bar-Zohar - Import</td>
      <td>4.6</td>
      <td>637.0</td>
      <td>8184958455</td>
      <td>340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>My Story</td>
      <td>by Kamala Das</td>
      <td>4.5</td>
      <td>42.0</td>
      <td>8172238975</td>
      <td>178</td>
    </tr>
  </tbody>
</table>
</div>




```python
amazon['amazon_price'].isnull().sum() #checking for null values in price column
```




    0




```python
amazon.shape
```




    (709, 6)




```python
#amazon.isnull().sum()
```


```python
amazon.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_isbn-10</th>
      <th>amazon_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>709.000000</td>
      <td>709.000000</td>
      <td>7.090000e+02</td>
      <td>709.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.870240</td>
      <td>277.279267</td>
      <td>7.726101e+09</td>
      <td>263.012694</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.466478</td>
      <td>747.859566</td>
      <td>2.696328e+09</td>
      <td>155.363593</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.250006e+09</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>23.000000</td>
      <td>8.131733e+09</td>
      <td>154.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>67.000000</td>
      <td>8.192911e+09</td>
      <td>227.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>196.000000</td>
      <td>9.352704e+09</td>
      <td>344.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>6566.000000</td>
      <td>9.960900e+09</td>
      <td>895.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
flipkart.drop_duplicates(inplace=True)
flipkart.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>flipkart_author</th>
      <th>flipkart_isbn10</th>
      <th>flipkart_title</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sidney Sheldon</td>
      <td>8172234902</td>
      <td>TELL ME YOUR DREAMS</td>
      <td>902</td>
      <td>209</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>1862305277</td>
      <td>The Boy in the Striped Pyjamas</td>
      <td>83</td>
      <td>372</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anil Lamba</td>
      <td>9350294311</td>
      <td>ROMANCING THE BALANCE SHEET</td>
      <td>352</td>
      <td>477</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bar-Zohar Michael</td>
      <td>8184958455</td>
      <td>Mossad</td>
      <td>560</td>
      <td>280</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kamala Das</td>
      <td>8172238975</td>
      <td>MY STORY</td>
      <td>322</td>
      <td>178</td>
      <td>4.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
flipkart.shape
```




    (1382, 6)




```python
flipkart['flipkart_price'].isnull().sum() #checking for null values in price column
```




    0




```python
flipkart.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>flipkart_isbn10</th>
      <th>flipkart_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.382000e+03</td>
      <td>1382.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.758529e+09</td>
      <td>263.436324</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.643135e+09</td>
      <td>210.164006</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.250006e+09</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.172235e+09</td>
      <td>149.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.192911e+09</td>
      <td>220.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.352865e+09</td>
      <td>320.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.960900e+09</td>
      <td>5201.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Joining both the tables by using ISBN number


```python
books=pd.merge(amazon,flipkart,how='outer',left_on='amazon_isbn-10',right_on='flipkart_isbn10')
books=books.drop_duplicates()
```


```python
books.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amazon_title</th>
      <th>amazon_author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_isbn-10</th>
      <th>amazon_price</th>
      <th>flipkart_author</th>
      <th>flipkart_isbn10</th>
      <th>flipkart_title</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tell Me your Dreams</td>
      <td>by Sidney Sheldon</td>
      <td>4.4</td>
      <td>160.0</td>
      <td>8172234902</td>
      <td>209</td>
      <td>Sidney Sheldon</td>
      <td>8172234902</td>
      <td>TELL ME YOUR DREAMS</td>
      <td>902</td>
      <td>209</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Boy in the Striped Pyjamas (Definitions)</td>
      <td>by John Boyne</td>
      <td>4.6</td>
      <td>134.0</td>
      <td>1862305277</td>
      <td>350</td>
      <td></td>
      <td>1862305277</td>
      <td>The Boy in the Striped Pyjamas</td>
      <td>83</td>
      <td>372</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Romancing the Balance Sheet: For Anyone Who Ow...</td>
      <td>by Anil Lamba</td>
      <td>4.5</td>
      <td>156.0</td>
      <td>9350294311</td>
      <td>477</td>
      <td>Anil Lamba</td>
      <td>9350294311</td>
      <td>ROMANCING THE BALANCE SHEET</td>
      <td>352</td>
      <td>477</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Mossad</td>
      <td>by Michael Bar-Zohar - Import</td>
      <td>4.6</td>
      <td>637.0</td>
      <td>8184958455</td>
      <td>340</td>
      <td>Bar-Zohar Michael</td>
      <td>8184958455</td>
      <td>Mossad</td>
      <td>560</td>
      <td>280</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>My Story</td>
      <td>by Kamala Das</td>
      <td>4.5</td>
      <td>42.0</td>
      <td>8172238975</td>
      <td>178</td>
      <td>Kamala Das</td>
      <td>8172238975</td>
      <td>MY STORY</td>
      <td>322</td>
      <td>178</td>
      <td>4.3</td>
    </tr>
  </tbody>
</table>
</div>



We can drop some unnecessary columns now.


```python
books.drop(columns=['flipkart_author','flipkart_title','flipkart_isbn10'],inplace=True)
```


```python
books.rename({
    'amazon_title':'Book Title',
    'amazon_author':'Author',
    'amazon_isbn-10':'ISBN No.'
},axis=1,inplace=True)
books.set_index('ISBN No.',inplace=True)

```


```python
books.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book Title</th>
      <th>Author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_price</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
    </tr>
    <tr>
      <th>ISBN No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8172234902</th>
      <td>Tell Me your Dreams</td>
      <td>by Sidney Sheldon</td>
      <td>4.4</td>
      <td>160.0</td>
      <td>209</td>
      <td>902</td>
      <td>209</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>1862305277</th>
      <td>The Boy in the Striped Pyjamas (Definitions)</td>
      <td>by John Boyne</td>
      <td>4.6</td>
      <td>134.0</td>
      <td>350</td>
      <td>83</td>
      <td>372</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>9350294311</th>
      <td>Romancing the Balance Sheet: For Anyone Who Ow...</td>
      <td>by Anil Lamba</td>
      <td>4.5</td>
      <td>156.0</td>
      <td>477</td>
      <td>352</td>
      <td>477</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>8184958455</th>
      <td>Mossad</td>
      <td>by Michael Bar-Zohar - Import</td>
      <td>4.6</td>
      <td>637.0</td>
      <td>340</td>
      <td>560</td>
      <td>280</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>8172238975</th>
      <td>My Story</td>
      <td>by Kamala Das</td>
      <td>4.5</td>
      <td>42.0</td>
      <td>178</td>
      <td>322</td>
      <td>178</td>
      <td>4.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
books[books['flipkart_price']>1000] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book Title</th>
      <th>Author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_price</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
    </tr>
    <tr>
      <th>ISBN No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1409346455</th>
      <td>Photography: The Definitive Visual History</td>
      <td>by Tom Ang</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>2</td>
      <td>1581</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>1506247598</th>
      <td>5 Lb. Book of Gre Practice Problems (Manhattan...</td>
      <td>by Manhattan Prep</td>
      <td>4.6</td>
      <td>151.0</td>
      <td>1</td>
      <td></td>
      <td>1799</td>
      <td></td>
    </tr>
    <tr>
      <th>1260142655</th>
      <td>CISSP All-in-One Exam Guide, Eighth Edition</td>
      <td>by Shon Harris - Import</td>
      <td>3.7</td>
      <td>11.0</td>
      <td>4</td>
      <td></td>
      <td>5201</td>
      <td></td>
    </tr>
    <tr>
      <th>9352704339</th>
      <td>Self-Assessment &amp; Review Medicine (Part A &amp; B)...</td>
      <td>by Mudit Khanna</td>
      <td>3.3</td>
      <td>16.0</td>
      <td>1</td>
      <td>33</td>
      <td>1235</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>9351524167</th>
      <td>SRB’s Manual of Surgery</td>
      <td>by Sriram Bhat M</td>
      <td>3.9</td>
      <td>33.0</td>
      <td>1</td>
      <td>58</td>
      <td>1145</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



There are certain outliers that need to be removed as the costliest book in amazon dataset is only 895 Rs.


```python
remove=books[books['flipkart_price']>1000].index.tolist()
books.drop(remove,inplace=True)
```

### Analysis


```python
plt.figure(figsize=(10, 6))
sns.distplot(books['flipkart_price'], color='#047BD5', label="Flipkart",kde=False, bins=30,)
sns.distplot(books['amazon_price'], color='#FF9900', label="Amazon",kde=False, bins=30)


plt.legend()
plt.xlabel('Price')
plt.ylabel('Number of Books')
```




    Text(0, 0.5, 'Number of Books')




![png](Images/output_28_1.png)


From the distribution of the prices on the two websites, we can see that there are some very distinct patterns starting with the spike in the range of 160-180 Rs

Creating two other columns for further study.


```python
books['Price_difference']=books['flipkart_price']-books['amazon_price']
books['percent_difference']=-(books['Price_difference'])/books['flipkart_price']
```


```python
books.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book Title</th>
      <th>Author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_price</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
      <th>Price_difference</th>
      <th>percent_difference</th>
    </tr>
    <tr>
      <th>ISBN No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8172234902</th>
      <td>Tell Me your Dreams</td>
      <td>by Sidney Sheldon</td>
      <td>4.4</td>
      <td>160.0</td>
      <td>209</td>
      <td>902</td>
      <td>209</td>
      <td>4.5</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1862305277</th>
      <td>The Boy in the Striped Pyjamas (Definitions)</td>
      <td>by John Boyne</td>
      <td>4.6</td>
      <td>134.0</td>
      <td>350</td>
      <td>83</td>
      <td>372</td>
      <td>4.5</td>
      <td>22</td>
      <td>-0.059140</td>
    </tr>
    <tr>
      <th>9350294311</th>
      <td>Romancing the Balance Sheet: For Anyone Who Ow...</td>
      <td>by Anil Lamba</td>
      <td>4.5</td>
      <td>156.0</td>
      <td>477</td>
      <td>352</td>
      <td>477</td>
      <td>4.5</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8184958455</th>
      <td>Mossad</td>
      <td>by Michael Bar-Zohar - Import</td>
      <td>4.6</td>
      <td>637.0</td>
      <td>340</td>
      <td>560</td>
      <td>280</td>
      <td>4.5</td>
      <td>-60</td>
      <td>0.214286</td>
    </tr>
    <tr>
      <th>8172238975</th>
      <td>My Story</td>
      <td>by Kamala Das</td>
      <td>4.5</td>
      <td>42.0</td>
      <td>178</td>
      <td>322</td>
      <td>178</td>
      <td>4.3</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
books['Price_difference'].describe()
```




    count    708.000000
    mean       4.427966
    std       57.390266
    min     -218.000000
    25%       -9.000000
    50%        0.000000
    75%       24.000000
    max      367.000000
    Name: Price_difference, dtype: float64



We see is a price difference maximum of 367 and a minimum of -218.
Lets find these books.


```python
books[books['Price_difference']==books['Price_difference'].max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book Title</th>
      <th>Author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_price</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
      <th>Price_difference</th>
      <th>percent_difference</th>
      <th>min_price</th>
    </tr>
    <tr>
      <th>ISBN No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8131733661</th>
      <td>Hughes Electrical and Electronic Technology, 10e</td>
      <td>by Hughes</td>
      <td>4.2</td>
      <td>30.0</td>
      <td>328</td>
      <td>189</td>
      <td>695</td>
      <td>4.4</td>
      <td>367</td>
      <td>-0.528058</td>
      <td>328</td>
    </tr>
  </tbody>
</table>
</div>




```python
books[books['Price_difference']==books['Price_difference'].min()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book Title</th>
      <th>Author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_price</th>
      <th>flipkart_ratings count</th>
      <th>flipkart_price</th>
      <th>flipkart_stars</th>
      <th>Price_difference</th>
      <th>percent_difference</th>
      <th>min_price</th>
    </tr>
    <tr>
      <th>ISBN No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1409323498</th>
      <td>The Animal Book: A Visual Encyclopedia of Life...</td>
      <td>by DK</td>
      <td>4.8</td>
      <td>19.0</td>
      <td>781</td>
      <td>5</td>
      <td>563</td>
      <td>4.6</td>
      <td>-218</td>
      <td>0.387211</td>
      <td>563</td>
    </tr>
  </tbody>
</table>
</div>



Hence, We see that flipkart is charging 111.8 % extra compared to Amazon on "Hughes Electrical and Electronic Technology".
while Amazon is charging 38.7% extra on "The Animal Book: A Visual Encyclopedia of Life"
Yeah, absorb that !!

#### Distortion Plots

We can see more clearly that the mean is shifter away from 0 when the books that are at the same price are excluded. The graph clearly shows that flipkart is mostly charging more for the same book.


```python
plt.figure(figsize=(8, 5))
sns.distplot(books['Price_difference'][books['Price_difference']!=0], color='orange', bins=80)
plt.xlabel('Price Difference')
```




    Text(0.5, 0, 'Price Difference')




![png](Images/output_40_1.png)


A large number of books have same price.


```python
plt.figure(figsize=(10, 6))
sns.distplot(books['percent_difference'], color='blue', bins=80)
plt.xlabel('Price Difference (Percentage)')
```




    Text(0.5, 0, 'Price Difference (Percentage)')




![png](Images/output_42_1.png)


#### The Frugal Mind
No! finding a cheaper price is not being stingy, Its called being smart


```python
books['min_price']=None
for i in books.index:
    if books['amazon_price'][i]<=books['flipkart_price'][i]:
        books['min_price'][i]=books['amazon_price'][i]
    else:
        books['min_price'][i]=books['flipkart_price'][i]
        

```


```python
print('Therefore an individual if buys all the books in the dataset only from Flipkart would spend Rs.{}'.format(books['flipkart_price'].sum()))
print('Whereas an individual if buys all the books in the dataset only from Amazon would spend Rs.{}\nHence saving Rs.{} on purchase'.format(books['amazon_price'].sum(),books['flipkart_price'].sum()-books['amazon_price'].sum()))
```

    Therefore an individual if buys all the books in the dataset only from Flipkart would spend Rs.190490
    Whereas an individual if buys all the books in the dataset only from Amazon would spend Rs.187355
    Hence saving Rs.3135 on purchase
    

#### So Amazon is the cheapest option ?


```python
print('Instead of buying only from amazon, a person buying smartly from both websites would save Rs. {} \non purchase of all books in the dataset'.format(books['amazon_price'].sum()-books['min_price'].sum()))
```

    Instead of buying only from amazon, a person buying smartly from both websites would save Rs. 10346 
    on purchase of all books in the dataset
    


```python
cart=[('Flipkart Buyer',books['flipkart_price'].sum()),('Amazon Buyer',books['amazon_price'].sum()),('Smart Buyer',books['min_price'].sum())]
cart_df=pd.DataFrame(cart,columns=['Buyer','Spending'])
```


```python
ax = sns.barplot(x='Buyer', y="Spending", data=cart_df)
```


![png](Images/output_49_0.png)



```python
no_diff = books[books['Price_difference'] == 0].count()[0]/books.shape[0]
print('Percentage of books which are priced the same: {0:.2f}%'.format(no_diff*100))

amz = books[books['Price_difference'] > 0].count()[0]/books.shape[0]
print('Better priced on Amazon: {0:.2f}%'.format(amz*100))

flp = books[books['Price_difference'] < 0].count()[0]/books.shape[0]
print('Better priced on Flipkart: {0:.2f}%'.format(flp*100))
```

    Percentage of books which are priced the same: 25.14%
    Better priced on Amazon: 45.06%
    Better priced on Flipkart: 29.80%
    

#### How much would one overpay if buying only from


```python
print('Flipkart: {0:.2f}%'.format((100*(books['flipkart_price'].sum() - books['min_price'].sum())/books['min_price'].sum())))
print()
print('Amazon: {0:.2f}%'.format((100*(books['amazon_price'].sum() - books['min_price'].sum())/books['min_price'].sum())))
```

    Flipkart: 7.62%
    
    Amazon: 5.84%
    


```python
plt.figure(figsize=(10, 6))
sns.kdeplot(books['amazon_price'], color='#FF9900')
sns.kdeplot(books['flipkart_price'], color='#047BD5')
sns.kdeplot(books['min_price'], color='green', label='Least Price')
```




    <AxesSubplot:>




![png](Images/output_53_1.png)


Clearly Amazon wins this battle of books.

## Part 2 - Wordcloud Sentiment Analysis

We always come across situations when we wish to buy a product but are unable to decide whether to buy or not because its not rated 4+ stars but then its also not rated 1 star. So we start reading some reviews but then there are thousands of them !! Whom to trust ??

In this part of the analysis we shall be classifying reviews as positive, negative or neutral by means of VADER classification tool by calculating compound score for each review.

For that we will be selecting an average rated book with review count more than 1000 from the above used dataset. 


```python
amazon=amazon.drop_duplicates()
amazon.isnull().sum()
```




    amazon_title            0
    amazon_author           0
    amazon_rating           4
    amazon_reviews count    4
    amazon_isbn-10          0
    amazon_price            0
    dtype: int64




```python
amazon=amazon.dropna()
amazon=amazon.drop_duplicates()
```


```python
amazon['amazon_rating'].unique()
```




    array(['4.4', '4.6', '4.5', '4.1', '3.5', '3.8', '4.2', '3.7', '4.3',
           '4.7', '3.9', '5.0', '4.8', '4.0', '4.9', '2.0', '3.6',
           '4.5 out of 5 stars', '4.3 out of 5 stars', '4.6 out of 5 stars',
           '4.2 out of 5 stars', '4.4 out of 5 stars', '4.1 out of 5 stars',
           '4.0 out of 5 stars', '4.7 out of 5 stars', '3.6 out of 5 stars',
           '5.0 out of 5 stars', '1.4', '3.3', '3.0', '1.0', '3.4', '3.1',
           nan, '3.2', '2.4'], dtype=object)



Ratings need to be converted to float data type


```python
rating_n=[]
for i in amazon['amazon_rating']:
    if type(i)==str:
        if len(i)>3:
            rating_n.append(i[0:3])
        else:
            rating_n.append(i)
    else:
        rating_n.append(i)
amazon['amazon_rating']=[float(x) for x in rating_n] 
```


```python
amazon['amazon_reviews count']=amazon['amazon_reviews count'].str.replace(r',','')            

amazon['amazon_reviews count']=[float(x) for x in amazon['amazon_reviews count']] 
       
```

Applying our search criteria to find a book for review analysis.


```python
amazon[(amazon['amazon_rating']<4) & (amazon['amazon_rating']>3.5) & (amazon['amazon_reviews count']>1000)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amazon_title</th>
      <th>amazon_author</th>
      <th>amazon_rating</th>
      <th>amazon_reviews count</th>
      <th>amazon_isbn-10</th>
      <th>amazon_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Making India Awesome: New Essays and Columns</td>
      <td>by Chetan Bhagat</td>
      <td>3.8</td>
      <td>1188.0</td>
      <td>8129137429</td>
      <td>100</td>
    </tr>
    <tr>
      <th>129</th>
      <td>One Indian Girl</td>
      <td>Chetan Bhagat</td>
      <td>3.6</td>
      <td>4929.0</td>
      <td>8129142147</td>
      <td>96</td>
    </tr>
  </tbody>
</table>
</div>



###### Unfortunately we'll have to chose a Chetan Bhagat book

### Making India Awesome

For this analysis the review data of this book has been scraped using Chrome scraper extention.

Importing required libraries


```python
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings("ignore")
```


```python
review = pd.read_csv('indiacb.csv')
review.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Asin</th>
      <th>UserName</th>
      <th>Rating</th>
      <th>Subject</th>
      <th>ReviewDate</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8129137429</td>
      <td>Ovn</td>
      <td>2.0 out of 5 stars</td>
      <td>\n\n\n\n\n\n\n\n  \n  \n    The only reason wh...</td>
      <td>Reviewed in India on 7 October 2017</td>
      <td>\n\n\n\n\n\n\n\n\n\n  \n  \n    \n  Let's see ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8129137429</td>
      <td>amz cus</td>
      <td>1.0 out of 5 stars</td>
      <td>\n\n\n\n\n\n\n\n  \n  \n    first this is not ...</td>
      <td>Reviewed in India on 4 May 2018</td>
      <td>\n\n\n\n\n\n\n\n\n\n  \n  \n    \n  chetan bag...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8129137429</td>
      <td>Varun Shah</td>
      <td>5.0 out of 5 stars</td>
      <td>\n\n\n\n\n\n\n\n  \n  \n    A weakipedia of wh...</td>
      <td>Reviewed in India on 22 June 2018</td>
      <td>\n\n\n\n\n\n\n\n\n\n  \n  \n    \n  . Making I...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8129137429</td>
      <td>janmejay thakker</td>
      <td>2.0 out of 5 stars</td>
      <td>\n\n\n\n\n\n\n\n  \n  \n    He is a good story...</td>
      <td>Reviewed in India on 26 December 2015</td>
      <td>\n\n\n\n\n\n\n\n\n\n  \n  \n    \n  He is a go...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8129137429</td>
      <td>Amazon Customer</td>
      <td>4.0 out of 5 stars</td>
      <td>\n\n\n\n\n\n\n\n  \n  \n    Must read book for...</td>
      <td>Reviewed in India on 27 January 2020</td>
      <td>\n\n\n\n\n\n\n\n\n\n  \n  \n    \n  Very nice ...</td>
    </tr>
  </tbody>
</table>
</div>



### Cleaning up the data


```python
for i in range(len(review)):
    review['Rating'][i]=int(review['Rating'][i][0])
    

s=review['Subject'].str.findall(r'[a-z0-9A-Z]+')
v=[]
for i in s:
    v.append(' '.join(i))
review['Subject']=v 


s=review['Review'].str.findall(r'[a-z0-9A-Z]+')
v=[]
for i in s:
    v.append(' '.join(i))
review['Review']=v 

s=review['ReviewDate'].str.findall(r'(\d{1,2})?[ -]?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\.\,]*[ -](\d{4})')
v=[]
for i in s:
    v.append(' '.join(i[0]))
review['ReviewDate']=v  
    
```


```python
review.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Asin</th>
      <th>UserName</th>
      <th>Rating</th>
      <th>Subject</th>
      <th>ReviewDate</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8129137429</td>
      <td>Ovn</td>
      <td>2</td>
      <td>The only reason why I bought this was because ...</td>
      <td>7 Oct 2017</td>
      <td>Let s see what I could have spend my bad inves...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8129137429</td>
      <td>amz cus</td>
      <td>1</td>
      <td>first this is not a novel</td>
      <td>4 May 2018</td>
      <td>chetan bagat i bought this after reading 2stat...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8129137429</td>
      <td>Varun Shah</td>
      <td>5</td>
      <td>A weakipedia of what ails us and how to cure o...</td>
      <td>22 Jun 2018</td>
      <td>Making India Awesome is an exceptional manual ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8129137429</td>
      <td>janmejay thakker</td>
      <td>2</td>
      <td>He is a good storyteller but has been left no ...</td>
      <td>26 Dec 2015</td>
      <td>He is a good storyteller but has been left no ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8129137429</td>
      <td>Amazon Customer</td>
      <td>4</td>
      <td>Must read book for every teenager</td>
      <td>27 Jan 2020</td>
      <td>Very nice book good compilation and thought pr...</td>
    </tr>
  </tbody>
</table>
</div>



Better!!


```python

# VADER sentiment analysis tool for getting Compound score.
def sentimental(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    return score

# VADER sentiment analysis tool for getting pos, neg and neu.
def sentimental_Score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'
```


```python
review['Sentiment_Score']=review['Review'].apply(lambda x: sentimental_Score(x))

```


```python
pos = review.loc[review['Sentiment_Score'] == 'pos']
neg = review.loc[review['Sentiment_Score'] == 'neg']
```


```python
def stemming(tokens):
    ps=nltk.WordNetLemmatizer()
    stem_words=[]
    for x in tokens:
        stem_words.append(ps.lemmatize(x))
    return stem_words
```


```python
def create_Word_Corpus(df):
    words_corpus = ''
    for val in df["Review"]:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = stemming(tokens)
        for words in tokens:
            words_corpus = words_corpus + words + ' '
    return words_corpus
```


```python
def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordclouds.png', facecolor='k', bbox_inches='tight')
```


```python
pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))
neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))
```


```python
plot_Cloud(pos_wordcloud)
```


![png](Images/output_83_0.png)



    <Figure size 432x288 with 0 Axes>



```python
plot_Cloud(neg_wordcloud)

```


![png](Images/output_84_0.png)



    <Figure size 432x288 with 0 Axes>



```python

```
