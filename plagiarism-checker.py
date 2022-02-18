import os.path
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import dot
from numpy.linalg import norm

class NotIntegerError(Exception):
    pass

# 문서를 불러와 단어로 토큰화 후, 단어들을 word_list에 저장후 word_list 반환
def doc_tokenize(doc_name):
    with open(doc_name, 'rt') as fp:
        string = fp.read()

    word_list = word_tokenize(string)

    # 유사도 계산시 정확성을 높이기 위해 큰 의미가 없는 단어인 불용어를 word_list에서 제거
    word_list = [word for word in word_list if word not in stop_words]

    # 소문자와 대문자로 인해 의미 구별이 되는 것을 방지하기 위해, 모든 단어를 소문자화
    word_list = [word.lower() if word.islower() == False else word for word in word_list]

    return word_list

# list안 word의 term frequency 값 계산 후 dict 형태로 반환
def tf(list):
    tf_dict = {word : list.count(word) if word in list else 0 for word in word_zip}

    return tf_dict

# list안 word의 tf 값과 idf 값을 곱하여 tf-idf 값 계산 후 알파벳 순으로 정렬하여 list 원소가 (word, tf-idf) 형식을 가진 list 형태로 반환
def tf_idf(list):
    tf_dict = tf(list)
    tf_idf_dict = {word : tf_dict[word] * idf_dict[word] for word in tf_dict.keys()}

    return sorted(tf_idf_dict.items())

# doc_1과 doc_2 문서의 cosine 유사도를 계산 후 유사도 값을 반환
def cos_similarity(doc_1_name, doc_2_name):
    # doc_1과 doc_2 문서의 tf-idf값 계산
    doc_1 = tf_idf(doc_tokenize(doc_1_name))
    doc_2 = tf_idf(doc_tokenize(doc_2_name))

    # doc_1의 word의 tf-idf 값을 vactor_1에 할당
    vector_1 = [value[1] for value in doc_1]

    # doc_2의 word의 tf-idf 값을 vactor_2에 할당
    vector_2 = [value[1] for value in doc_2]

    # vector_1과 vector_2 사이의 각도를 구한후 100을 곱하여 % 수치로 반환, 소숫점 2자리까지 반올림
    return round((dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2)))*100, 2)

while True:
    try:
        # 문서 수 입력
        doc_count = float(input('Please enter the count of documents : '))

        if doc_count % 1 != 0:
            raise NotIntegerError()

        doc_count = int(doc_count)
        doc_name_list = []

        i = 0
        while i < doc_count:
            doc_name = input(f'Please enter the name of documents [{i + 1}{"/"}{doc_count}] : ') + ".txt"

            # 존재하지 않은 문서 이름을 입력시 다시 입력, 존재하는 문서 입력시 doc_name_list에 할당
            if os.path.isfile(doc_name):
                doc_name_list.append(doc_name)
                i += 1
            else:
                print('Please enter the name of an existing document.')
        break
    except ValueError:
        # 문서 수를 입력할 때 숫자를 입력하지 않으면 excpet 발생
        print('Please enter the number.')
    except NotIntegerError:
        # 문서 수를 입력할 때 정수를 입력하지 않으면 excpet 발생
        print('Please enter the integer.')

stop_words = set(stopwords.words('english'))

# idf 값을 계산하기 위해 모든 문서를 doc_zip에 할당
doc_zip = [doc_tokenize(name) for name in doc_name_list]

# tf-idf 값을 계산하기 위해 모든 문서의 단어를 중복되지 않게 word_zip에 할당
word_zip = list(set([word for doc in doc_zip for word in doc]))

# 각 단어마다 inverse document frequency 값 계산 후 dict에 할당
idf_dict = {}
for word in word_zip:
    word_count = 0
    for doc in doc_zip:
        if word in doc:
            word_count += 1
    idf_dict[word] = np.log((1 + doc_count) / (word_count))

# 경로 상의 모든 문서의 서로 간의 유사도를 계산 후 similarity_dict에 저장
similarity_dict = {(doc_name_list[i], doc_name_list[j]) : cos_similarity(doc_name_list[i], doc_name_list[j]) for i in range(len(doc_name_list)-1) for j in range(i+1, doc_count)}

# 유사도가 가장 큰 문서 2개를 계산 후 출력
key_min = max(similarity_dict.keys(), key = lambda x: similarity_dict[x])
value_min = max(similarity_dict.values())

print(f"The similarity between {key_min[0]} and {key_min[1]} is highest at {value_min}%")