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
 
### 1.3 Thuật toán ML
 
<p align="left">Các thuật toán ML chính là cốt lõi của MLlib. Chúng bao gồm các thuật toán học tập phổ biến như phân loại, hồi quy, phân cụm và lọc cộng tác.

MLlib chuẩn hóa các API để giúp kết hợp nhiều thuật toán vào một đường dẫn hoặc quy trình làm việc dễ dàng hơn. Các khái niệm chính là API đường ống, trong đó khái niệm đường ống được lấy cảm hứng từ dự án scikit-learning.

+ <b>Transformer</b>: đây là một thuật toán biển đổi một Dataframe thành một Dataframe khác. Về mặt lý thuyết nó thực hiện một phương thức transform() dùng để chuyển đỏi một Dataframe thành một Dataframe khác bằng cách thêm một hoặc nhiều cột.
+ <b>Estimator</b>: là một thuật toán phù hợp trên Dataframe để tạo Transformer. Về mặt kỹ thuật, Estimator triển khai phương thức fit() và chấp nhận DataFrame tạo ra một mô hình, là một transformer.
 
 
