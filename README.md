# SYCL-matrix-multiplication
基于oneAPI的C++/SYCL程序来执行矩阵乘法操作

`dpcpp dxr.cpp dxr_ndrange_var.cpp -o dxr -w -O3`



|d|m|v|p|
|-|-|-|-|
|matrix number to be multiplied|size of work_group|verify output with linear computation|print output matrix|

e.g.
`dxr -d 3 -m 16 -v -p`