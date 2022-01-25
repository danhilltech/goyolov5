#ifndef __GOYOLO_API_H__
#define __GOYOLO_API_H__
#include <stdint.h>

#ifdef __cplusplus
thread_local char *torch_last_err = nullptr;

extern "C"
{
    typedef torch::Tensor *tensor;
    typedef torch::jit::script::Module *module;
#define PROTECT(x)                         \
    try                                    \
    {                                      \
        x                                  \
    }                                      \
    catch (const exception &e)             \
    {                                      \
        torch_last_err = strdup(e.what()); \
    }
#else
typedef void *tensor;
typedef void *module;
#endif

    char *get_and_reset_last_err();
    tensor at_new_tensor();
    module atm_load_on_device(char *, int device);
    void init_module(module m);
    void init_module_half(module m);
    void infer(module m, int device, void *data, int size, int half, void *dst);
    int cudaDeviceCount();
    size_t at_dim(tensor t);
    void at_shape(tensor t, int64_t *dims);
    tensor at_get(tensor t, int index);
    int getBatchSize(tensor detections);
    int getNumClasses(tensor detections);
    void detections_info(tensor t);
    void at_free(tensor t);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif
