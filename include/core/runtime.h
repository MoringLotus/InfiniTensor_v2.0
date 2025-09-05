#pragma once
#ifndef RUNTIME_H
#define RUNTIME_H
#include "core/kernel.h"
#include "core/graph.h"
#include <infiniop/handle.h>
#include <infinirt.h>
#include <memory>
#include <thread>
#include <mutex>

namespace infini
{
  inline infiniDevice_t CURRENT_DEVICE_TYPE = INFINI_DEVICE_CPU;
  inline int CURRENT_DEVICE_ID = 0;
  struct ContextObj
  {
    infiniDevice_t device = CURRENT_DEVICE_TYPE;
    int deviceId = CURRENT_DEVICE_ID;
    infinirtStream_t stream = nullptr;
  };
  using Context = Ref<ContextObj>;

  class RuntimeObj
  {
  private:
    // 全局 map: thread_id -> Context
    std::unordered_map<std::thread::id, Context> threadContexts;
    std::mutex mtx; // 保护 map

    RuntimeObj() {}

  public:
    RuntimeObj(const RuntimeObj &) = delete;
    RuntimeObj &operator=(const RuntimeObj &) = delete;

    // 每个线程唯一的 Runtime
    static Runtime &getInstance();

    // 每个线程初始化自己的 Context
    void initThreadContext(infiniDevice_t device, int deviceId);

    // 获取活跃 Context
    Context getCurrentThreadContext() const;
    void setCurrentDevice(infiniDevice_t device, int deviceId);

    static void init();
    static void getAllDeviceCount(int *count_array);
    void run(const Graph &graph) const;
    void *allocHost(size_t size);
    void *allocDevice(size_t size);
    void deallocHost(void *ptr);
    void deallocDevice(void *ptr);
    void memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
    void memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);
    void *mallocAsync(size_t size, infinirtStream_t stream);
    void freeAsync(void *ptr, infinirtStream_t stream);
    void synchronize() const;
    // void *getWorkspace(size_t size) const;

    // string toString() const;
  };
} // namespace infini
#endif // RUNTIME_H
