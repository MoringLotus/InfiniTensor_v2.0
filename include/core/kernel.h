#pragma once
#include "core/operator.h"
#include <infinirt.h>
namespace infini
{

    using KernelAttrs = std::tuple<infiniDevice_t, OpType::underlying_t>;
    class Kernel
    {
    public:
        Kernel() {}
        virtual ~Kernel() {}
        virtual void compute(const Operator &op, infinirtStream_t stream) const = 0;
    };

    class KernelRegistry
    {
    public:
        using KernelRecord = tuple<Kernel *const, const string, const int>; // Kernel, name, ID

    private:
        std::map<KernelAttrs, KernelRecord> kernels;
        int nKernels = 0;

    public:
        ~KernelRegistry()
        {
            for (auto &[k, v] : kernels)
                delete std::get<0>(v);
        }
        static KernelRegistry &getInstance()
        {
            static KernelRegistry instance;
            return instance;
        }
        bool registerKernel(const KernelAttrs &key, Kernel *kernel, string name)
        {
            IT_ASSERT(kernels.find(key) == kernels.end(),
                      "Kernel already registered");
            kernels.emplace(key, KernelRecord{kernel, name, ++nKernels});
            return true;
        }
        Kernel *getKernel(const KernelAttrs &kernelAttrs) const
        {
            auto it = kernels.find(kernelAttrs);
            IT_ASSERT(it != kernels.end(), "Kernel not found");
            return std::get<0>(it->second);
        }
        const KernelRecord &getKernelItem(const KernelAttrs &kernelAttrs) const
        {
            return kernels.at(kernelAttrs);
        }
    };

} // namespace infini

#define _REGISTER_KERNEL_1(device, opType, kernel, name, cnt)                 \
    namespace infini                                                          \
    {                                                                         \
        static const bool _CAT(_register_kernel_, cnt) =                      \
            KernelRegistry::getInstance().registerKernel(KernelAttrs{device,  \
                                                                     opType}, \
                                                         new kernel(), name); \
    }

#define REGISTER_KERNEL(device, opType, kernel, name) \
    _REGISTER_KERNEL_1(device, opType, kernel, name, __COUNTER__)
