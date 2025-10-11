#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include "core/blob.h"
#include "core/object.h"
#include "core/dtype.h"
#include "utils/utils.h"

namespace infini
{

    class TensorObj : public Object
    {
        friend class GraphObj;

    private:
        Fuid fuid;
        DataType dtype;
        Shape shape;
        Stride stride;
        Blob data = nullptr;
        vector<WRef<OperatorObj>> targets;
        WRef<OperatorObj> source;

    public:
        TensorObj(Shape shape, DataType dtype);
        TensorObj(Shape shape, Stride stride, DataType dtype);
        virtual ~TensorObj() {}

        // =============Get TensorObj attributes=================
        UidBaseType getFuid() const;
        DataType getDataType() const;
        Shape getShape() const;
        void setShape(Shape shape_);
        Stride getStride() const;
        void setStride(Stride stride_);
        Blob getData() const;
        ElementType getElement() const;
        ElementType getStorageSize() const;
        ElementType getTotalBytes() const;
        ElementType getRank() const;
        OpVec getTargets() const;
        Operator getSource() const;

        string toString() const override;
        // ============= TensorObj Data Operations==============
        void setData(void *data_);
        void dataMalloc(const Runtime &runtime);

        template <typename T>
        T getRawDataPtr() const
        {
            static_assert(std::is_pointer_v<T>,
                          "Raw data pointer has a type of pointer");
            IT_ASSERT(data != nullptr);
            return data->getPtr<T>();
        }

        void printData(const Runtime &runtime, size_t maxElements = 0, int precision = 4) const;

    private:
        // ============= Change Graph Operations==============
        void addTarget(const Operator &op);
        void setSource(const Operator &op);
        void removeTarget(const Operator &op);
        Stride computeContiguousStride(const Shape &shape) const;
        bool checkValid() const;

        template <typename T>
        void printDataImpl(const Runtime &runtime, size_t maxElements = 0, int precision = 4) const;
    };

} // namespace infini
#endif // TENSOR_H
