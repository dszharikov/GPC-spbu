#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "linear-algebra.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1)
{
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1 - t0).count();
    if (dt == 0)
    {
        return 0;
    }
    return ((n + n + n) * sizeof(float) * 1e-9) / (dt * 1e-6);
}

void print(const char *name, std::array<duration, 5> dt, std::array<double, 2> bw)
{
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i = 0; i < 5; ++i)
    {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i = 0; i < 2; ++i)
    {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names()
{
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_vector_times_vector(int n, OpenCL &opencl)
{
    auto a = random_vector<float>(n);
    auto b = random_vector<float>(n);
    Vector<float> result(n), expected_result(n);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "vector_times_vector");
    auto t0 = clock_type::now();
    vector_times_vector(a, b, expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, result.size() * sizeof(float));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_result);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
    opencl.queue.flush();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_result, begin(result), end(result));
    auto t4 = clock_type::now();
    verify_vector(expected_result, result);
    print("vector-times-vector",
          {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
          {bandwidth(n + n + n, t0, t1), bandwidth(n + n + n, t2, t3)});
}

void profile_matrix_times_vector(int n, OpenCL &opencl)
{
    auto a = random_matrix<float>(n, n);
    auto b = random_vector<float>(n);
    Vector<float> result(n), expected_result(n);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "matrix_times_vector");
    auto t0 = clock_type::now();
    matrix_times_vector(a, b, expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, result.size() * sizeof(float));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_result);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
    opencl.queue.flush();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_result, begin(result), end(result));
    auto t4 = clock_type::now();
    verify_vector(expected_result, result, 1e-1f);
    print("matrix-times-vector",
          {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
          {bandwidth(n * n + n + n, t0, t1), bandwidth(n * n + n + n, t2, t3)});
}

#define BLOCK_SIZE 16
void profile_matrix_times_matrix(int n, OpenCL &opencl)
{
    auto a = random_matrix<float>(n, n);
    auto b = random_matrix<float>(n, n);
    Matrix<float> result(n, n), expected_result(n, n);

    // ---- OpenMP baseline ----
    auto t0 = clock_type::now();
    matrix_times_matrix(a, b, expected_result);
    auto t1 = clock_type::now();

    // ---- OpenCL copy-in ----
    cl::Kernel kernel(opencl.program, "matrix_times_matrix");
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, result.size() * sizeof(float));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_result);

    opencl.queue.flush();
    auto t2 = clock_type::now();

    // ---- OpenCL kernel ----
    // Важно: используем 2D NDRange(n, n).
    // Можно задать локальную размерность, например (256, 256), если n кратно 256.
    // Ниже для универсальности localRange оставляем пустым (NullRange),
    // но в самом ядре всё равно используем тильный метод.
    opencl.queue.enqueueNDRangeKernel(kernel,
                                      cl::NullRange,
                                      cl::NDRange(n, n),
                                      cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));
    opencl.queue.flush();
    auto t3 = clock_type::now();

    // ---- OpenCL copy-out ----
    cl::copy(opencl.queue, d_result, begin(result), end(result));
    auto t4 = clock_type::now();

    // Проверка результата
    verify_matrix(expected_result, result, 1e-1f);

    print("matrix-times-matrix",
          {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
          {bandwidth(n * n + n * n + n * n, t0, t1),
           bandwidth(n * n + n * n + n * n, t2, t3)});
}

void opencl_main(OpenCL &opencl)
{
    using namespace std::chrono;
    print_column_names();
    profile_vector_times_vector(1024 * 1024 * 10, opencl);
    profile_matrix_times_vector(1024 * 10, opencl);
    profile_matrix_times_matrix(1024, opencl);
}

const std::string src = R"(
kernel void vector_times_vector(global float* a,
                                global float* b,
                                global float* result) {
    const int i = get_global_id(0);
    result[i] = a[i] * b[i];
}

kernel void matrix_times_vector(global const float* a,
                                global const float* b,
                                global float* result) {
    const int i = get_global_id(0);
    const int n = get_global_size(0);
    float sum = 0;
    for (int j=0; j<n; ++j) {
        sum += a[i*n + j]*b[j];
    }
    result[i] = sum;
}

// Реализуем умножение матриц A(n,n) * B(n,n) = C(n,n) с использованием локальной памяти.
// Для простоты возьмём размер блока BLOCK_SIZE=16 (можно менять).
// Если n не кратно 256, возможны "условные" края, но здесь для демонстрации ядра
// оставим общий подход.
#define BLOCK_SIZE 16
kernel void matrix_times_matrix(global const float* a,
                                global const float* b,
                                global float* result)
{
    // Глобальные индексы
    int globalRow = get_global_id(0); // строка в C
    int globalCol = get_global_id(1); // столбец в C

    // Размер матрицы (по условию квадратная: n x n)
    int n = get_global_size(0);

    // Локальные индексы
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    // Локальная (shared) память
    __local float Ashared[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bshared[BLOCK_SIZE][BLOCK_SIZE];

    // Аккумулятор для текущего элемента (globalRow, globalCol)
    float sum = 0.0f;

    // Шагаем по блокам размера BLOCK_SIZE.
    // Каждый блок загружаем в локальную память.
    // ceil(n/BLOCK_SIZE) = (n + BLOCK_SIZE - 1)/BLOCK_SIZE
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        // Колонка блока для A
        int tiledCol = t * BLOCK_SIZE + localCol;
        // Строка блока для B
        int tiledRow = t * BLOCK_SIZE + localRow;

        // Загружаем в локальную память, проверяя границы
        if (globalRow < n && tiledCol < n) {
            Ashared[localRow][localCol] = a[globalRow * n + tiledCol];
        } else {
            Ashared[localRow][localCol] = 0.0f;
        }

        if (tiledRow < n && globalCol < n) {
            Bshared[localRow][localCol] = b[tiledRow * n + globalCol];
        } else {
            Bshared[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Перемножаем подблоки
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += Ashared[localRow][k] * Bshared[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Записываем результат
    // (проверяем, что не выходим за границы, если n не кратно BLOCK_SIZE)
    if (globalRow < n && globalCol < n) {
        result[globalRow * n + globalCol] = sum;
    }
}
)";

int main()
{
    try
    {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            {CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try
        {
            program.build(devices);
        }
        catch (const cl::Error &err)
        {
            for (const auto &device : devices)
            {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    }
    catch (const cl::Error &err)
    {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
                  << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
