//==============================================================
// Matrix Multiplication: SYCL ND-range private mem
//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>

using namespace sycl;

void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N0, size_t N1, size_t N2, size_t M)
{
    std::cout << "Configuration         : MATRIX_SIZE= " << N0 << "x" << N2 << " | WORK_GROUP_SIZE= " << M << "x" << M << "\n";

    // # Create buffers for matrices
    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);

    // # Submit command groups to execute on device
    auto e = q.submit([&](handler &h)
                      {
        //# Create accessors to copy buffers to the device
        auto A = a.get_access<access::mode::read>(h);
        auto B = b.get_access<access::mode::read>(h);
        auto C = c.get_access<access::mode::write>(h);

        //# Define size for ND-Range and work-group size
        range<2> global_size(N0,N2);
        range<2> work_group_size(M,M);

        //# Parallel Compute Matrix Multiplication
        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            //# Use private mem to store intermediate result
            float temp = 0.f;
            for (int k = 0; k < N1; k++) {
                temp += A[i*N1+k] * B[k*N2+j];
            }
            C[i*N2+j] = temp;
        }); });
    c.get_access<access::mode::read>();

    // # print kernel compute duration from event profiling
    auto kernel_duration = (e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds\n";
}
