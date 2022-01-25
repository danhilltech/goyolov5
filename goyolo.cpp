#include <torch/torch.h>
#include <torch/script.h>
#include <stdexcept>
#include <vector>
#include "goyolo.h"

using namespace std;
enum Det
{
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

char *get_and_reset_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

at::Device device_of_int(int d)
{
    if (d < 0)
        return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

tensor at_new_tensor()
{
    PROTECT(return new torch::Tensor();)
    return nullptr;
}

module atm_load_on_device(char *filename, int device)
{
    PROTECT(
        return new torch::jit::script::Module(
                   torch::jit::load(filename, device_of_int(device)));)
    return nullptr;
}

void init_module(module m)
{
    PROTECT(
        m->eval();)
}

void init_module_half(module m)
{
    PROTECT(
        m->to(torch::kHalf);
        m->eval();)
}

void infer(module m, int device, void *data, int size, int half, void *dst)
{
    torch::NoGradGuard no_grad;

    PROTECT(
        auto tensor_img = torch::from_blob(data, {1, size, size, 3}, torch::kByte).to(device_of_int(device));
        tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous(); // BHWC -> BCHW (Batch, Channel, Height, Width)

        if (half == 0)
        {
            tensor_img = tensor_img.to(torch::kFloat32).div(255);
        } else
        {
            tensor_img = tensor_img.to(torch::kFloat16).div(255);
        }

        std::vector<torch::jit::IValue>
            inputs;
        inputs.emplace_back(tensor_img);
        torch::jit::IValue output = m->forward(inputs);

        auto detections = output.toTuple()->elements()[0].toTensor().to(torch::kFloat32).cpu().contiguous();

        memcpy(dst, detections.data_ptr(), detections.numel() * detections.element_size());

        return;)
    return;
}

struct Rect
{

    int x;
    int y;
    int width;
    int height;
};

struct Point
{

    int x;
    int y;
};

struct Detection
{
    Rect bbox;
    float score;
    int class_idx;
};

int getBatchSize(tensor detections)
{
    PROTECT(return detections->size(0);)
    return -1;
}

int getNumClasses(tensor detections)
{
    PROTECT(constexpr int item_attr_size = 5;
            return detections->size(2) - item_attr_size;)
    return -1;
}

int cudaDeviceCount()
{
    PROTECT(
        return torch::cuda::device_count();)
    return -1;
}

void detections_info(tensor t)
{
    cout << "dims: " << t->dim() << endl;
    cout << "sizes: " << t->sizes() << endl;
    cout << "batch size: " << t->size(0) << endl;
    cout << "PREDS size: " << t->size(1) << endl;
    cout << "num classes: " << t->size(2) - 5 << endl;
}

size_t at_dim(tensor t)
{
    PROTECT(return t->dim();)
    return -1;
}

void at_shape(tensor t, int64_t *dims)
{
    PROTECT(int i = 0; for (int64_t dim
                            : t->sizes()) dims[i++] = dim;)
}

tensor at_get(tensor t, int index)
{
    PROTECT(return new torch::Tensor((*t)[index]);)
    return nullptr;
}

void at_free(tensor t) { delete (t); }
