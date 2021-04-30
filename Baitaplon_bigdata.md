# Bigdata
# 1. Machine learning trong PySpark
## 1.1 Giới thiệu về Spark Mllib
<p align="left">Spark MLlib được sử dụng để thực hiện học máy trong Apache Spark. MLlib bao gồm các thuật toán và tiện ích phổ biến. MLlib trong Spark là một thư viện mở rộng của học máy để thảo luận về các thuật toán chất lượng cao và tốc độ cao.</p>
<p align="center"> <img src ="https://bigdata-etl.com/wp-content/uploads/2019/02/spark-mllib-logo.png" />
<p align="center"> Giới thiệu về Spark Mllib </p>
<p align="left">Chúng ta có thể dề cập một vài thuật toán được sử dụng trong Spark Mllib được liệt kê bên dưới như sau:</p>

<p align="left">
  
+ <b>Mllib.classification</b>: Đây là các thuật toán thuôc nhánh phân loại mà chúng ta có thể kể đến như phân loại nhị phân, phân loại đa lớp và phân tích hồi quy. Một số thuật toán phổ biến nhất trong phân loại là random forest, Naïve Bayes, Decesion Tree,…

+ <b>Mllib.clustering</b>: Clustering là một vấn đề học không giám sát, theo đó nhằm mục đích nhóm các tập con của các thực thể với nhau dựa trên một số khái niệm về sự giống nhau.

+ <b>Mllib.fpm</b>: Frequent Pattern Mining (hay còn gọi là đối sánh mẫu thường xuyên) là khai thác các mục thường xuyên, tập phổ biến, chuỗi con hoặc các cấu trúc con khác thường nằm trong số các bước đầu tiên để phân tích một tập dữ liệu lớn.

+ <b>Mllib.linalg</b>: Các tiện ích MLlib đối với đại số tuyến tính.

+ <b>Mllib.recommendation</b>: Lọc cộng tác thường được sử dụng cho các hệ thống gợi ý. Các kỹ thuật này nhằm mục đích điền vào các mục còn thiếu của ma trận với liên kết mục người dùng.

+ <b>Spark.mllib</b>: Hiện hỗ trợ lọc cộng tác dựa trên mô hình, trong đó người dùng và sản phẩm được mô tả bằng một tập hợp nhỏ các yếu tố tiềm ẩn có thể được sử dụng để dự đoán các mục nhập bị thiếu. spark.mllib sử dụng thuật toán Bình phương tối thiểu xen kẽ (ALS – Alternating Least Square) để tìm hiểu các yếu tố tiềm ẩn này.

+ <b>Mllib.regression</b>: Hồi quy tuyến tính thuộc nhánh thuật toán hồi quy. Mục tiêu của hồi quy là tìm mối quan hệ và sự phụ thuộc giữa các biến. Giao diện làm việc với mô hình hồi quy tuyến tính và tóm tắt mô hình tương tự như trường hợp hồi quy logistic.
  
 </p>
 
## 1.2 Công cụ sử dụng Spark.Mllib

<p align="left">Spark.Mllib là API học máy chính cho Spark. Thư viện Spark.Mllib cung cấp một API cấp cao hơn được xây dựng trên DataFrames để xây dựng các pipeline cho machine learning.

Các Công cụ Spark Mllib như:

+ Thuật toán ML

+ Featurization

+ Pipelines

+ Sự ổn định

+ Tiện ích

 </p>
 
### 1.2.1 Thuật toán ML
 
<p align="left">Các thuật toán ML chính là cốt lõi của MLlib. Chúng bao gồm các thuật toán học tập phổ biến như phân loại, hồi quy, phân cụm và lọc cộng tác.

MLlib chuẩn hóa các API để giúp kết hợp nhiều thuật toán vào một đường dẫn hoặc quy trình làm việc dễ dàng hơn. Các khái niệm chính là API đường ống, trong đó khái niệm đường ống được lấy cảm hứng từ dự án scikit-learning.

+ <b>Transformer</b>: đây là một thuật toán biển đổi một Dataframe thành một Dataframe khác. Về mặt lý thuyết nó thực hiện một phương thức transform() dùng để chuyển đỏi một Dataframe thành một Dataframe khác bằng cách thêm một hoặc nhiều cột.
+ <b>Estimator</b>: là một thuật toán phù hợp trên Dataframe để tạo Transformer. Về mặt kỹ thuật, Estimator triển khai phương thức fit() và chấp nhận DataFrame tạo ra một mô hình, là một transformer.
 
### 1.2.2 Featurization
<p align="left">Featurization bao gồm trích xuất, biến đổi, giảm kích thước và lựa chọn:
 
 + Tính năng trích xuất sẽ được trích xuất từ dữ liệu thô.
 
 + Tính năng biến đổi bao gồm mở rộng, tái tạo và chỉnh sửa.
 
 + Tính năng lựa chọn liên quan đến việc chọn một tập hợp con các tính năng cần thiết từ một tập hợp lớn các tính năng.
 
</p>

### 1.2.3 Pipelines

<p align="left">Pipelines giúp kết nối các Estimator và Transformer lại với nhau theo một quy trình của làm việc của ML. Đồng thời nó cũng cung cấp công cụ để đánh giá, xây dựng và điều chỉnh ML pipelines.</p>

### 1.2.4 Sự ổn định
<p align="left">Sự ổn định giúp duy trì quá trình tính toán các thuật toán, mô hình và Pipelines. Giúp chúng ta giảm thiểu được chi phí và tái sử dụng.</p>

### 1.2.5 Tiện ích
<p align="left">Các tiện ích cho đại số tuyến tính, thống kê và xử lý dữ liệu. Ví dụ mllib.linalg hỗ trợ cho đại số tuyến tính.</p>

## 1.3 Sử dụng Spark Mllib với Python
### Phân loại nhị phân

<p align="left">Ví dụ sau đây sẽ hướng dẫn chúng ta load tập dữ liệu, xây dựng mô hình hồi quy Logistic và đưa ra dự đoán kết quả mô hình để tính toán lỗi huấn luyện.</p>

```objective-python

from pyspark.mllib.classification import LogisticRegressionWithSGD
from numpy import array

# Load và phân tích data

data = sc.textFile("mllib/data/sample_svm_data.txt")

parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

model = LogisticRegressionWithSGD.train(parsedData)

# Xây dựng mô hình

labelsAndPreds = parsedData.map(lambda point: (int(point.item(0)),
        model.predict(point.take(range(1, point.size)))))

# Đánh gia mô hình trên tập dữ liệu train

trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

print("Training Error = " + str(trainErr))

```

### Hồi quy tuyến tính
<p align="left">Ví dụ sau sẽ minh họa cách load dữ liệu training, phân tích cú pháp nó dưới dạng RDD của LabeledPoint. Sau đó chúng ta sẽ sử dụng LinearRegressionWithSGD để xây dựng một mô hình tuyến tính đơn giản để dự đoán các giá trị label. Chúng ta sẽ tính toán sai số trung bình bình phương (Mean Squared Error) ở cuối để đánh giá mức độ phù hợp (goodness of fit)</p>

```objective-python
from pyspark.mllib.regression import LinearRegressionWithSGD
from numpy import array

# Load and phân tích data
data = sc.textFile("mllib/data/ridge-data/lpsa.data")
parsedData = data.map(lambda line: array([float(x) for x in line.replace(',', ' ').split(' ')]))

# Xây dựng mô hình
model = LinearRegressionWithSGD.train(parsedData)

# Đánh giá mô hình trên tập dữ liệu train
valuesAndPreds = parsedData.map(lambda point: (point.item(0),
        model.predict(point.take(range(1, point.size)))))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y)/valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))

```
