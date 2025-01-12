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
#include <cmath> // std::abs

#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration   = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1 - t0).count();
    if (dt == 0) { return 0; }
    return ((n + n + n) * sizeof(float) * 1e-9) / (dt * 1e-6);
}

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
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

struct OpenCL {
    cl::Platform   platform;
    cl::Device     device;
    cl::Context    context;
    cl::Program    program;
    cl::CommandQueue queue;
};

// ------------------ Профилирование редукции (reduce) ------------------
void profile_reduce(int n, OpenCL& opencl) {
    // Исходный вектор
    auto a = random_vector<float>(n);
    // Результат для OpenMP
    float expected_result = 0.f;
    // Результат для OpenCL
    float result = 0.f;

    // ---- OpenMP baseline ----
    auto t0 = clock_type::now();
    expected_result = reduce(a);
    auto t1 = clock_type::now();

    // ---- OpenCL copy-in ----
    cl::Buffer d_in(opencl.queue, begin(a), end(a), true);
    // Только одно число в выходном буфере (скаляр result).
    cl::Buffer d_out(opencl.context, CL_MEM_READ_WRITE, sizeof(float));
    // Инициализируем нулями
    float zero = 0.0f;
    opencl.queue.enqueueWriteBuffer(d_out, CL_TRUE, 0, sizeof(float), &zero);

    opencl.queue.flush();
    auto t2 = clock_type::now();

    // ---- OpenCL kernel (многоступенчатая редукция на GPU) ----
    cl::Kernel kernel_pass(opencl.program, "reduce_pass");
    cl::Buffer d_temp(opencl.context, CL_MEM_READ_WRITE, n*sizeof(float));

    const size_t BLOCK_SIZE = 256;
    size_t current_size = n;

    while (current_size > 1) {
        size_t num_groups = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernel_pass.setArg(0, d_in);
        kernel_pass.setArg(1, d_temp);
        kernel_pass.setArg(2, (int)current_size);
        kernel_pass.setArg(3, cl::Local(BLOCK_SIZE * sizeof(float)));

        opencl.queue.enqueueNDRangeKernel(kernel_pass,
                                          cl::NullRange,
                                          cl::NDRange(num_groups*BLOCK_SIZE),
                                          cl::NDRange(BLOCK_SIZE));

        // swap
        std::swap(d_in, d_temp);
        current_size = num_groups;
    }

    // Финализирующее ядро, копирующее d_in[0] -> d_out[0]
    cl::Kernel kernel_finalize(opencl.program, "reduce_finalize");
    kernel_finalize.setArg(0, d_in);
    kernel_finalize.setArg(1, d_out);
    opencl.queue.enqueueNDRangeKernel(kernel_finalize,
                                      cl::NullRange,
                                      cl::NDRange(1),
                                      cl::NullRange);

    opencl.queue.flush();
    auto t3 = clock_type::now();

    // ---- OpenCL copy-out ----
    opencl.queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(float), &result);
    auto t4 = clock_type::now();

    // Проверка (вывод разницы)
    float diff = std::abs(expected_result - result);
    std::cout << "[reduce] CPU=" << expected_result 
              << ", GPU=" << result 
              << ", diff=" << diff << std::endl;

    print("reduce",
          {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
          {
              bandwidth(n, t0, t1),     // OpenMP bandwidth
              bandwidth(n, t2, t3)      // OpenCL kernel bandwidth
          });
}

// ------------------ Профилирование сканирования (scan inclusive) ------------------
void profile_scan_inclusive(int n, OpenCL& opencl) {
    auto arr = random_vector<float>(n);
    Vector<float> vec_opencl(arr), vec_omp(arr);

    // Сначала CPU scan
    opencl.queue.flush();
    cl::Kernel kernel_sc(opencl.program, "scan_inclusive");
    cl::Kernel kernel_fin(opencl.program, "finish_scan_inclusive");

    auto t0 = clock_type::now();
    scan_inclusive(vec_omp);
    auto t1 = clock_type::now();

    // 1) copy-in
    cl::Buffer buf_in(opencl.queue, begin(arr), end(arr), false);
    opencl.queue.finish();
    auto t2 = clock_type::now();

    // 2) kernel
    kernel_sc.setArg(0, buf_in);
    kernel_sc.setArg(1, buf_in);
    kernel_sc.setArg(2, 1);
    opencl.queue.enqueueNDRangeKernel(kernel_sc,
                                  cl::NullRange,
                                  cl::NDRange(n),
                                  cl::NDRange(1024));

    kernel_sc.setArg(0, buf_in);
    kernel_sc.setArg(1, buf_in);
    kernel_sc.setArg(2, 1*1024);
    opencl.queue.enqueueNDRangeKernel(kernel_sc,
                                  cl::NullRange,
                                  cl::NDRange(n/1024),
                                  cl::NDRange(1024));

    kernel_sc.setArg(0, buf_in);
    kernel_sc.setArg(1, buf_in);
    kernel_sc.setArg(2, 1*1024*1024);
    opencl.queue.enqueueNDRangeKernel(kernel_sc,
                                  cl::NullRange,
                                  cl::NDRange(n/(1024*1024)),
                                  cl::NDRange(10));

    kernel_fin.setArg(0, buf_in);
    kernel_fin.setArg(1, 1024);

    opencl.queue.enqueueNDRangeKernel(kernel_fin,
                                  cl::NullRange,
                                  cl::NDRange((n/1024) - 1),
                                  cl::NullRange);

    opencl.queue.finish();
    std::cout << "→ GPU SCAN DONE.\n";

    auto t3 = clock_type::now();

    // 3) copy-out
    cl::copy(opencl.queue, buf_in, begin(vec_opencl), end(vec_opencl));
    auto t4 = clock_type::now();


    std::cout << "LastCheck → GPU=" << vec_opencl[n-1]
              << ", CPU=" << vec_omp[n-1]
              << ", diff=" << std::abs(vec_opencl[n-1] - vec_omp[n-1]) << '\n';

    print("scan-inclusive",
                  { t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3 },
                  { bandwidth(n*n + n*n + n*n, t0, t1),
                    bandwidth(n*n + n*n + n*n, t2, t3) });
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    // Запуск редукции
    profile_reduce(1024 * 1024 * 10, opencl);
    // Запуск сканирования
    profile_scan_inclusive(1024 * 1024 * 10, opencl);
}

// ------------------ Код ядер OpenCL ------------------
const std::string src = R"(
/* ------------------------ reduce -------------------------
   Ядро reduce_pass:
   - Считывает данные из in[ global_id ]
   - Использует локальную память partial[]
   - Делает свёртку внутри work-group
   - Записывает результат групп в out[ group_id ]
*/
kernel void reduce_pass(global const float* in,
                        global float* out,
                        int current_size,
                        local float* partial)
{
    int gid        = get_global_id(0);
    int lid        = get_local_id(0);
    int group_id   = get_group_id(0);
    int local_size = get_local_size(0);

    float val = 0.0f;
    if (gid < current_size) {
        val = in[gid];
    }
    partial[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Делаем редукцию в локальной памяти
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            partial[lid] += partial[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Записываем результат группы
    if (lid == 0) {
        out[group_id] = partial[0];
    }
}

/* reduce_finalize: просто out[0] = in[0] */
kernel void reduce_finalize(global const float* in,
                            global float* out)
{
    out[0] = in[0];
}

/* ------------------- scan_block -----------------------*/
kernel void scan_inclusive(global float* a, global float* result, int step) {
    int gID  = get_global_id(0);
    int lID  = get_local_id(0);
    int lSZ  = get_local_size(0);
    local float locBuf[1024];

    // читаем ...
    locBuf[lID] = a[(step - 1) + gID * step];
    barrier(CLK_LOCAL_MEM_FENCE);

    float accum = locBuf[lID];
    for (int offset = 1; offset < lSZ; offset <<= 1) {
        float temp = 0.0f;
        if (lID >= offset) {
            temp = locBuf[lID - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        accum += temp;
        locBuf[lID] = accum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    a[(step - 1) + gID * step] = locBuf[lID];
}

kernel void finish_scan_inclusive(global float* a, int stp) {
    int i = get_global_id(0);
    for (int x = 0; x < stp - 1; x++) {
        a[(i + 1)*stp + x] += a[(i + 1)*stp - 1];
    }
}
)"; // end of src string

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';

        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0 };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            std::cerr << "No devices in this context\n";
            return 1;
        }

        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';

        // Создаём программу
        cl::Program program(context, src);
        try {
            // соберём с -cl-std=CL1.2 
            program.build(devices, "-cl-std=CL1.2");
        } catch (const cl::Error& err) {
            for (const auto& dev : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << log;
            }
            throw;
        }

        cl::CommandQueue queue(context, device);

        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
                  << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
