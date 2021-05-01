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

### Phân cụm
<p align="left">Sau khi tải và phân tích dữ liệu, chúng ta sử dụng đối tượng KMeans để phân cụm dữ liệu thành hai cụm. Số lượng các cụm được chuyển đến thuật toán. Sau đó, chúng ta tính toán (Within Set Sum of Squared Error - WSSSE). Ta có thể giảm số đo sai số này bằng cách tăng k.</p>

```objective-pyhton
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt

# Load và phân tích data
data = sc.textFile("kmeans_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Xây dựng mô hình (phân cụm data)
clusters = KMeans.train(parsedData, 2, maxIterations=10,
        runs=30, initialization_mode="random")

# Đánh giá phân cụm dựa trên Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))
```
<p align="left">Ngoài ra, chúng ta cũng có thể sử dụng RidgeRegressionWithSGD hoặc LassoWithSGD và so sánh các lỗi trung bình bình phương (Mean Squared Error) khi huấn luyện.</p>

### Lọc cộng tác
<p align="left">Trong ví dụ sau, chúng ta sẽ load dữ liệu đánh giá. Mỗi dòng bao gồm một người dùng, một sản phẩm và một đánh giá. Chúng ta sử dụng ALS.train() mặc định, giả định xếp hạng là rõ ràng. Chúng ta đánh giá đề xuất bằng cách đo Sai số trung bình bình phương của dự đoán xếp hạng.</p>

```objective-python

from pyspark.mllib.recommendation import ALS
from numpy import array

# Load và phân tích data
data = sc.textFile("mllib/data/als/test.data")
ratings = data.map(lambda line: array([float(x) for x in line.split(',')]))

# Xây dựng mô hình gợi ý sử dụng Alternating Least Squares
model = ALS.train(ratings, 1, 20)

# Đánh giá mô hình trên tập dữ liệu train
testdata = ratings.map(lambda p: (int(p[0]), int(p[1])))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).reduce(lambda x, y: x + y)/ratesAndPreds.count()
print("Mean Squared Error = " + str(MSE))
```
<p align="left">Nếu ma trận xếp hạng được lấy từ một nguồn khác (tức là nó được suy ra từ các signal khác), chúng ta cũng có thể sử dụng phương pháp ngầm định đào tạo để có được kết quả tốt hơn.</p>

 # 2. Spark Dataframe
 ## 2.1 Khái niệm
 
Khung dữ liệu (dataframe) là một bảng hoặc cấu trúc giống như mảng hai chiều, trong mà mỗi cột chứa các phép đo trên một biến và mỗi hàng chứa một trường hợp.

Vì vậy, một DataFrame có siêu dữ liệu bổ sung do định dạng bảng của nó, cho phép Spark chạy một số tối ưu hóa nhất định trên truy vấn đã hoàn thành.

<p align="center"> <img src ="https://ongxuanhong.files.wordpress.com/2016/05/spark-dataframes.png" />

Mặt khác, RDD theo như chúng ta biết chỉ là một Resilient Distribution Dataset có nhiều hộp đen dữ liệu không thể được tối ưu hóa như các hoạt động có thể được thực hiện chống lại nó, không bị ràng buộc.

Tuy nhiên, chúng ta có thể chuyển từ DataFrame sang RDD thông qua phương thức rdd của nó và ngược lại có thể chuyển từ RDD sang DataFrame (nếu RDD ở định dạng bảng) thông qua phương thức toDF.

Nhìn chung, chúng ta nên sử dụng DataFrame trong trường hợp có thể do tối ưu hóa truy vấn tích hợp.

## 2.2 Một số tính năng của Dataframe và nguồn dữ liệu PySpark
### 2.2.1 Tính năng

DataFrame được phân phối trong tự nhiên, làm cho nó trở thành một cấu trúc dữ liệu có khả năng chịu lỗi và có tính khả dụng cao.

Đánh giá lười biếng là một chiến lược đánh giá giữ việc đánh giá một biểu thức cho đến khi giá trị của nó là cần thiết. Nó tránh đánh giá lặp lại. Đánh giá lười biếng trong Spark có nghĩa là quá trình thực thi sẽ không bắt đầu cho đến khi một hành động được kích hoạt. Trong Spark, bức tranh về sự lười biếng xuất hiện khi các phép biến đổi Spark xảy ra.
<p align="center"> <img src ="https://cdn.helpex.vn/upload/2019/2/19/ar/04-21-36-927-3156016a-bdfd-49ab-b9b1-a6878a618ac1.jpg" />

### 2.2.2 Nguồn dữ liệu PySpark

Dữ liệu có thể được tải vào thông qua tệp CSV, JSON, XML hoặc tệp Parquet. Nó cũng có thể được tạo bằng cách sử dụng RDD hiện có và thông qua bất kỳ cơ sở dữ liệu nào khác, như Hive hoặc Cassandra . Nó cũng có thể lấy dữ liệu từ HDFS hoặc hệ thống tệp cục bộ.

### 2.2.3 Một số lợi ích khi sử dụng Spark Dataframe

+ Xử lý dữ liệu có cấu trúc và bán cấu trúc: DataFrames được thiết kế để xử lý một tập hợp lớn dữ liệu có cấu trúc cũng như bán cấu trúc . Các quan sát trong Spark DataFrame được tổ chức dưới các cột được đặt tên, giúp Apache Spark hiểu được lược đồ của Dataframe. Điều này giúp Spark tối ưu hóa kế hoạch thực thi trên các truy vấn này. Nó cũng có thể xử lý hàng petabyte dữ liệu.

+ Slicing và Dicing: API DataFrames thường hỗ trợ các phương pháp phức tạp để cắt và phân loại dữ liệu. Nó bao gồm các hoạt động như "selecting" hàng, cột và ô theo tên hoặc theo số, lọc ra các hàng, v.v. Dữ liệu thống kê thường rất lộn xộn và chứa nhiều giá trị bị thiếu và không chính xác cũng như vi phạm phạm vi. Vì vậy, một tính năng cực kỳ quan trọng của DataFrames là quản lý rõ ràng dữ liệu bị thiếu.

+ Hỗ trợ nhiều ngôn ngữ: Hỗ trợ API cho các ngôn ngữ khác nhau như Python, R, Scala, Java, giúp những người có nền tảng lập trình khác nhau sử dụng dễ dàng hơn. 

+	Nguồn dữ liệu: DataFrames có hỗ trợ cho nhiều định dạng và nguồn dữ liệu, chúng ta sẽ xem xét vấn đề này sau trong hướng dẫn Pyspark DataFrames này. Họ có thể lấy dữ liệu từ nhiều nguồn khác nhau.

### 2.2.4 Hạn chế

<p align="center"> API DataFrame không hỗ trợ biên dịch thời gian một cách an toàn, điều này giới hạn người dùng khi thao tác dữ liệu khi không biết cấu trúc của dữ liệu.</p>

<p align="center">Ngoài ra, sau khi chuyển đổi đối tượng miền thành DataFrame, người dùng không thể tạo lại nó.</p>

# Tài liệu tham khảo
1. https://spark.apache.org/docs/latest/sql-programming-guide.html
2. https://towardsdatascience.com/the-most-complete-guide-to-pyspark-dataframes-2702c343b2e8
3. http://spark.apache.org/docs/latest/ml-guide.html
4. https://www.baeldung.com/spark-mlib-machine-learning
5. https://ichi.pro/vi/spark-for-machine-learning-su-dung-python-va-mllib-74075263465224
