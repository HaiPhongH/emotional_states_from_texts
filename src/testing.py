from crawl_data import CrawlingData
from phobert_model import TrainingBert

import pandas as pd
import joblib
import numpy as np

crawling = CrawlingData()

# good product
url = 'https://tiki.vn/dien-thoai-samsung-galaxy-note-20-ultra-8gb-256gb-hang-chinh-hang-p58616042.html?spid=58616044'

# bad product
url_2 = 'https://tiki.vn/dien-thoai-thtphone-a8-p9799424.html'

# crawl reviews from website
data_testing = crawling.crawl_data_with_url(url)
data_testing = pd.DataFrame(data_testing)
testing_file = 'data/test_data_good_product.csv'
crawling.save_to_csv(testing_file, data_testing)

# create model for feature extraction
model = TrainingBert()

print('Loading test data ...')
testing_review = []
with open(testing_file, encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:
    line = line.replace("\n", "")
    testing_review.append(model.standardize_data(line))

print('Creating PhoBERT features vectors ...')
test_features = model.create_feature_vector(testing_review)

# load pre-trained model
pretrained_model = joblib.load('src/pretrained_model.pkl')

# get the result for product
result = pretrained_model.predict(test_features)
print(result)

# analyse the result
bad_reviews = np.count_nonzero(result == 2)
normal_review = np.count_nonzero(result == 0)
good_review = np.count_nonzero(result == 1)

print ("Number of bad review is: ", bad_reviews)
print ("Number of normal review is: ", normal_review)
print ("Number of good review is: ", good_review)

if (good_review > bad_reviews) and (good_review > normal_review):
    print("The reviews are very good.")
elif (normal_review > good_review) and (normal_review > bad_reviews):
    print("The reviews are accecptable, you can check the product for more details.")
else:
    print("The reviews are really bad, be careful before buy this product.")


