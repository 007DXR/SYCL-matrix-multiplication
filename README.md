# SYCL-matrix-multiplication
## 基于oneAPI的C++/SYCL程序执行矩阵乘法操作
### 问题描述
编写⼀个基于oneAPI的C++/SYCL程序来执行矩阵乘法操作。需要考虑大尺寸矩阵的乘法操作以及不同线程之
间的数据依赖关系。通常在实现矩阵乘法时，可以使用块矩阵乘法以及共享内存来提高计算效率。
### 分析
利用基于SYCL的编程模型在GPU上实现矩阵乘法的计算，步骤如下：
1. 分配内存：在主机端分配内存空间用于存储输⼊矩阵和输出矩阵，同时在GPU端分配内存空间用于存储相应
的输入和输出数据。
2. 数据传输：将输入矩阵数据从主机端内存传输到GPU端内存中。
3. 核函数调用：在SYCL中，矩阵乘法的计算通常会在GPU上使用核函数来实现并行计算。核函数
会分配线程块和线程来处理不同的数据块。
4. 并行计算：在核函数中，每个线程负责计算输出矩阵的⼀个单独的元素。为了最大限度地利用
GPU的并行计算能力，通常会使用⼆维线程块和线程网格的方式来处理矩阵的乘法计算。
5. 数据传输：计算完成后，将输出矩阵数据从GPU端内存传输回主机端内存中，以便进⼀步处理或
分析。
在并行计算矩阵乘法时，可以利用线程块和线程的层次结构来优化计算。通过合理划分矩阵数据并利用共享内
存来减少全局内存访问的次数，可以⼤幅提高计算效率。此外，还可以利用GPU上的多个计算单元并执行行矩
阵乘法，进⼀步提高计算速度。

### 环境介绍
使用英特尔oneAPI Developer Cloud 服务。
访问[Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/)，在“Get Started”选项页面中， 点击页面最左下角 Connect with Jupyter* Lab 中的“Launch JupyterLab*”按钮直接启动 Jupypter 服务。

### 项目部署
进入文件夹 ”SYCL_Performance_Portability“
添加文件 [dxr.cpp](https://github.com/007DXR/SYCL-matrix-multiplication/blob/main/dxr.cpp) 和 [dxr_ndrange_var.cpp](https://github.com/007DXR/SYCL-matrix-multiplication/blob/main/dxr_ndrange_var.cpp)


### 测试

<!-- `pip install ipywidgets`

- 运行accelerator.py以选择要运行代码的目标设备

`run accelerator.py`
dpcpp lab/dxr.cpp lab/dxr_ndrange_var.cpp -o lab/dxr1 -w -O3

lab/dxr1 -d 3 -m 16 -v -p
 -->



- 编译文件


`dpcpp dxr.cpp dxr_ndrange_var.cpp -o dxr -w -O3`

编译代码dxr.cpp和dxr_ndrange_var.cpp，生成可执行文件dxr

- 运行可执行文件，参数有：d（必选）,m（必选）,v（可选）,p（可选）

`./dxr -d <DIMENSION> -m <WORK_GROUP_SIZE>  -v -p`


|d|m|v|p|
|-|-|-|-|
|矩阵维度（矩阵数量+1）|工作组大小|是否验证正确性|打印矩阵|


#### 示例
在终端中键入
`./dxr -d 3 -m 16 -v -p`

参数`-d 3 -m 16 -v -p`表示两个矩形，三个维度，工作组大小是16，验证正确性，打印矩阵。终端依次给出提示词：

`Dimension <DIMENSION_ID>`

`Matrix <MATRIX_ID>`

根据提示词输入数据，终端显示如下：

```
Dimension 0: 2
Dimension 1: 3
Dimension 2: 1

Matrix 0:
1 2 0
-1 0 2

Matrix 1:
1
0
-1
```


第1个矩阵的大小是2 X 3，第2个矩阵的大小是3 X 1，矩阵乘法的结果是
```
1
-3
```



<!-- - 选择设备

`./q xxx.sh <DEVICE_NAME>`

xxx.sh 需要包含形如`./dxr -d <DIMENSION> -m <WORK_GROUP_SIZE>  -v -p`的可执行文件

可供选择的设备有：
'GPU Gen9', 'GPU Iris XE Max', 'CPU Xeon 6128', 'CPU Xeon 8153' -->

<!-- #### 示例

! CHMOD 755 ./sample1.sh; ./q sample1.sh "GPU GEN9"

chmod 755 q; chmod 755 sample1.sh; ./q sample1.sh "GPU GEN9";
chmod 755 sample1.sh;  ./sample1.sh;
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1； 
chmod 755 sample1.sh;  ./sample1.sh; -->

