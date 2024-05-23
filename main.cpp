#include <iostream>
#include <CL/opencl.hpp>
#include <cstdio>
#include <cstdlib>

#define MATRIX_SIZE 1000
#define cl_assert(result) { if (result != CL_SUCCESS) \
    std::cout << "Assertion failed at line " << __LINE__ <<\
    " with code " << result << ": " << cl_code2string (result) << std::endl;}

void initialize_matrices(float* A, int N) {
    for (int i = 0; i < N * N; i++) {
//        A[i] = static_cast<float>(rand()) / (float)RAND_MAX;
//        B[i] = static_cast<float>(rand()) / (float)RAND_MAX;
        A[i] = i;
    }
}

std::string cl_code2string(int code) {
    switch (code) {
        case 0: return "CL_SUCCESS";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";

        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";

        default: return "see: /usr/include/CL/cl.h";

    }
}

int main(){
    cl_int errorcode = 0;

    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    for (int i = 0; i < all_platforms.size(); i++)
        std::cout << "Platform " << i <<  " : " +
                                          all_platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


    cl::Context context({default_device});

    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    std::string kernel_code=
            "__kernel void matrix_mul(const int M, const int N, const int K,"
            "                      __global const float* A,"
            "                      __global const float* B,"
            "                      __global float* C) {"
            "    const int globalRow = get_global_id(0);"
            "    const int globalCol = get_global_id(1);"
            "    float acc = 0.0f;"
            "    for (int k=0; k<K; k++) {"
            "        acc += A[k*M + globalRow] * B[globalCol*K + k];"
            "    }"
            "    C[globalCol*M + globalRow] = acc;"
            "}                                                                                                       ";
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    const int M = MATRIX_SIZE;
    const int N = MATRIX_SIZE;
    const int K = MATRIX_SIZE;
    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,MATRIX_SIZE*MATRIX_SIZE*sizeof(float), nullptr, &errorcode);
    cl_assert(errorcode);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,MATRIX_SIZE*MATRIX_SIZE*sizeof(float), nullptr, &errorcode);
    cl_assert(errorcode);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,MATRIX_SIZE*MATRIX_SIZE*sizeof(float), nullptr, &errorcode);
    cl_assert(errorcode);
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    // create matrix
    float *A = (float*) malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
    float *B = (float*) malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
    float *C = (float*) malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
    initialize_matrices(A, MATRIX_SIZE);
//    for(int i=0;i<MATRIX_SIZE*MATRIX_SIZE;i++) {
//        std::cout<<A[i]<<" ";
//    }
    std::cout<<std::endl;
    initialize_matrices(B, MATRIX_SIZE);
//    for(int i=0;i<MATRIX_SIZE*MATRIX_SIZE;i++) {
//        std::cout<<B[i]<<" ";
//    }
    std::cout<<std::endl;
    //write arrays A and B to the device
    errorcode = queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,MATRIX_SIZE*MATRIX_SIZE*sizeof(float),A);
    cl_assert(errorcode);
    errorcode = queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,MATRIX_SIZE*MATRIX_SIZE*sizeof(float),B);
    cl_assert(errorcode);

    //run the kernel
    cl::Kernel kernel_matrix_mul=cl::Kernel(program,"matrix_mul");
    errorcode = kernel_matrix_mul.setArg(0,M);
    cl_assert(errorcode);
    errorcode = kernel_matrix_mul.setArg(1,N);
    cl_assert(errorcode);
    errorcode = kernel_matrix_mul.setArg(2,K);
    cl_assert(errorcode);
    errorcode = kernel_matrix_mul.setArg(3,buffer_A);
    cl_assert(errorcode);
    errorcode = kernel_matrix_mul.setArg(4,buffer_B);
    cl_assert(errorcode);
    errorcode = kernel_matrix_mul.setArg(5,buffer_C);
    cl_assert(errorcode);
    const int TS = 16;
    cl::NDRange global(M, N);
    cl::NDRange local(TS, TS);

//    queue.enqueueNDRangeKernel(kernel_matrix_mul,NULL,global,local);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    errorcode = queue.enqueueNDRangeKernel(kernel_matrix_mul, NULL,global,cl::NullRange);
    cl_assert(errorcode);
    errorcode = queue.finish();
    cl_assert(errorcode);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Time difference = " << ms << " ms " << std::endl << std::endl;

    //read result C from the device to array C
    errorcode = queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,M*N*sizeof(float),C);
    cl_assert(errorcode);
//    std::cout<<" result: \n";
//    for(int i=0;i<MATRIX_SIZE*MATRIX_SIZE;i++) {
//        std::cout<<C[i]<<" ";
//    }
    free(A);
    free(B);
    free(C);
    return 0;
}