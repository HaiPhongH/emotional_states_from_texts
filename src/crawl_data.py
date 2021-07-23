from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd

class CrawlingData():

    def crawl_data_with_url(self, url):
        # user Chrome driver to get the data from a website
        driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
        print("Loading url: ", url)
        driver.get(url)
        
        list_review = []
        
        tiki = 'https://tiki.vn/'
        lazada = 'https://www.lazada.vn/'

        # colnames = ['review', 'label']
        # df = pd.DataFrame(columns = colnames)

        # number of review pages
        number_of_pages = 0
        while number_of_pages < 10:
            try:
                # if the url is from tiki
                if tiki in url:
                    # selenium will wait for maximum 20 seconds to find the element <div class = "review-comment">
                    WebDriverWait(driver, 20).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "div.review-comment")))

                # if the url is from lazada
                elif lazada in url:
                    # selenium will wait for maximum 20 seconds to find the element <div class = "item">
                    WebDriverWait(driver,20).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR,"div.item")))
                else:
                    print('Only crawl data from 2 websites, tiki.vn and lazada.vn.')
                    break
            except:
                print('There is no comment.')
                break

            # if the url is from tiki
            if tiki in url:
                # find all the reviews of the producs
                product_reviews = driver.find_elements_by_css_selector("div.review-comment")
            
            # if the url is from lazada
            else:
                product_reviews = driver.find_elements_by_css_selector("div.item")
            
            # for each product element in product reviews element
            # take only content element
            for product in product_reviews:
                if tiki in url:
                    review = product.find_element_by_css_selector ("div.review-comment__content").text
                else:
                    review = product.find_element_by_css_selector("div.content").text
                if (review != "" or review.strip()):
                    # print(review, "\n")

                    list_review.append(review)

                    # new_review = []
                    # new_review.append(review)
                    # new_review.append(0)
                    # row = pd.Series(new_review, index = df.columns)
                    # df = df.append(row, ignore_index = True)

            # go to the next page of review    
            try:
                if tiki in url:
                    button_next = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "a.btn.next")))
                else:
                    button_next = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "button.next-pagination-item.next")))
                driver.execute_script("arguments[0].click();", button_next)
                # print("next page")
                time.sleep(2)
                number_of_pages += 1
            except:
                print('Load serveral page')
                break
        driver.close()
        # return df
        return list_review

    # save data to csv file
    def save_to_csv(self, filename, dataframe):
        dataframe.to_csv(filename, index = False, header = False)

# url = 'https://tiki.vn/dien-thoai-iphone-12-pro-max-128gb-hang-chinh-hang-p70771651.html'
# url_1 = 'https://www.lazada.vn/products/laptop-gaming-gia-re-laptop-choi-game-asus-gl552jx-i5-4200h-vga-roi-nvidia-gtx-950m-4g-laptop-asus-laptop-gaming-i409206641-s3683690507.html?spm=a2o4n.searchlistcategory.list.42.1c884578WMdh8e&search=1'
# url_2 = 'https://tiki.vn/dien-thoai-thtphone-f7-p5325313.html'
# url_3 = 'https://tiki.vn/ao-croptop-form-rong-in-hoa-tiet-lon-nuoc-cuc-dang-yeu-p58719710.html'
# url_4 = 'https://tiki.vn/ao-croptop-nu-3-lo-theu-buom-chat-cotton-min-cuc-xinh-p58673657.html'
# url_5 = 'https://tiki.vn/combo-3-tinh-dau-senselab-nhap-khau-an-do-tinh-dau-sa-chanh-10ml-tinh-dau-oai-huong-10ml-tinh-dau-bac-ha-10ml-tinh-dau-thien-nhien-nguyen-chat-p52139397.html'

# crawling = CrawlingData()

# df = crawling.crawl_data_with_url(url_5)
# df.to_csv('data/data1.csv', mode = 'a', index = False, header = False)

# data = pd.read_csv('data/data1.csv')
# data = pd.DataFrame(data)
# data = data.values.tolist()
# positive = 0
# neutral = 0
# negative = 0

# for d in data:
#     if d[-1] == 0:
#         neutral += 1
#     elif d[-1] == 1:
#         positive += 1
#     else:
#         negative += 1

# print("pos: ", positive, ", neutral: ", neutral, ", negative: ", negative)

import underthesea

string = "Sản phẩm này rất tuyệt vời, tôi khuyên các bạn nên mua nó ngay khi có thể."
sw = []
with open("data/stopwords.txt", encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    sw.append(line.replace("\n",""))

new_string = underthesea.word_tokenize(string)
print(new_string)
filtered_line = [word for word in new_string if not word in sw]
print(filtered_line)
