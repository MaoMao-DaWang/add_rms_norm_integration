#include <ATen/ATen.h>
#include <c10/core/GradMode.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "launch.h"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& gamma,
    double epsilon, bool modelCompatible)
{
    TORCH_CHECK(x1.device().type() == c10::DeviceType::PrivateUse1, "x1 must be on NPU");
    TORCH_CHECK(x2.device() == x1.device() && gamma.device() == x1.device(), "devices must match");
    TORCH_CHECK(x1.dim() >= 2 && x1.dim() <= 4 && x1.numel() > 0, "expected nonempty rank 2..4");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1/x2 shapes must match");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type() && x1.scalar_type() == gamma.scalar_type(),
        "all input dtypes must match");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous() && gamma.is_contiguous(),
        "all inputs must be contiguous");
    const int64_t cols = x1.size(-1);
    TORCH_CHECK(cols >= 1 && cols <= 16384, "hidden size must be in [1,16384]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == cols, "gamma must have shape [hidden]");
    const int64_t rows = x1.numel() / cols;
    TORCH_CHECK(rows <= 1048576 && x1.numel() <= 33554432, "input exceeds supported size");
    const float eps = static_cast<float>(epsilon);
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0f, "epsilon must be positive and FP32-representable");
    TORCH_CHECK(!(c10::GradMode::is_enabled() &&
        (x1.requires_grad() || x2.requires_grad() || gamma.requires_grad())),
        "forward inference only: use torch.inference_mode(); backward is not implemented");
    int32_t dtype = -1;
    if (x1.scalar_type() == at::kFloat) dtype = 0;
    if (x1.scalar_type() == at::kHalf) dtype = 1;
    if (x1.scalar_type() == at::kBFloat16) dtype = 2;
    TORCH_CHECK(dtype >= 0, "only float32/float16/bfloat16 are supported");

    const c10_npu::NPUGuard guard(x1.device());
    // Query once per thread/device; never initialize ACL or create a private stream.
    static thread_local int32_t cachedDevice = -1;
    static thread_local int64_t cachedCores = 0;
    if (cachedDevice != x1.device().index()) {
        int64_t cores = 0;
        const auto status = aclrtGetDeviceInfo(x1.device().index(), ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
        TORCH_CHECK(status == ACL_SUCCESS && cores > 0, "cannot query vector core count: ", status);
        cachedDevice = x1.device().index();
        cachedCores = cores;
    }
    auto y = at::empty(x1.sizes(), x1.options());
    auto residual = at::empty(x1.sizes(), x1.options());
    auto statShape = x1.sizes().vec();
    statShape.pop_back();
    auto rstd = at::empty(statShape, x1.options().dtype(at::kFloat));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    arm_launch(x1.data_ptr(), x2.data_ptr(), gamma.data_ptr(),
        y.data_ptr(), rstd.data_ptr(), residual.data_ptr(),
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), dtype,
        static_cast<uint32_t>(cachedCores), eps, modelCompatible ? 1 : 0, stream);
    return {y, rstd, residual};
}
} // namespace

TORCH_LIBRARY(training_camp, m)
{
    m.def("add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-6, "
          "bool model_compatible=False) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(training_camp, PrivateUse1, m)
{
    m.impl("add_rms_norm", TORCH_FN(add_rms_norm));
}
