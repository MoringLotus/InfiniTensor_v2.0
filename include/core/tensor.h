#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include "core/blob.h"
#include "core/object.h"
#include "core/dtype.h"

namespace infini
{

    using ShapeElem = size_t;
    using Shape = vector<ShapeElem>;
    using StrideElem = ptrdiff_t;
    using Stride = vector<StrideElem>;
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
        ElementType getBytes() const;
        ElementType getRank() const;
        OpVec getTargets() const;
        Operator getSource() const;

        string toString() const override;
        // ============= TensorObj Data Operations==============
        void dataMalloc(const Blob &data_);
        void dataMalloc(const Runtime &runtime);

        template <typename T>
        T getRawDataPtr() const
        {
            static_assert(std::is_pointer_v<T>,
                          "Raw data pointer has a type of pointer");
            IT_ASSERT(data != nullptr);
            return data->getPtr<T>();
        }

    private:
        // ============= Change Graph Operations==============
        void addTarget(const Operator &op);
        void setSource(const Operator &op);
        void removeTarget(const Operator &op);
        Stride computeContiguousStride(const Shape &shape) const;
        bool checkValid() const;
    };

} // namespace infini
#endif // TENSOR_H
