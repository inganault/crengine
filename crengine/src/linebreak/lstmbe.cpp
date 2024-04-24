// Modified from https://github.com/unicode-org/icu/blob/0e4c2d8bc68bbd46f2b74c0404e0cc26a98251f7/icu4c/source/common/lstmbe.cpp

// © 2021 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "lstmbe.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Uncomment the following #define to debug.
// #define LSTM_DEBUG 1
// #define LSTM_VECTORIZER_DEBUG 1

#if LSTM_DEBUG || LSTM_VECTORIZER_DEBUG
#   include <stdio.h>
#endif
#if LSTM_DEBUG
#   include <assert.h>
#   define U_ASSERT(exp) assert(exp)
#else
#   define U_ASSERT(exp)
#endif

/**
 * Interface for reading 1D array.
 */
class ReadArray1D {
public:
    virtual ~ReadArray1D();
    virtual int32_t d1() const = 0;
    virtual float get(int32_t i) const = 0;

#ifdef LSTM_DEBUG
    void print() const {
        printf("\n[");
        for (int32_t i = 0; i < d1(); i++) {
           printf("%0.8e ", get(i));
           if (i % 4 == 3) printf("\n");
        }
        printf("]\n");
    }
#endif
};

ReadArray1D::~ReadArray1D()
{
}

/**
 * Interface for reading 2D array.
 */
class ReadArray2D {
public:
    virtual ~ReadArray2D();
    virtual int32_t d1() const = 0;
    virtual int32_t d2() const = 0;
    virtual float get(int32_t i, int32_t j) const = 0;
};

ReadArray2D::~ReadArray2D()
{
}

/**
 * A class to index a float array as a 1D Array without owning the pointer or
 * copy the data.
 */
class ConstArray1D : public ReadArray1D {
public:
    ConstArray1D() : data_(nullptr), d1_(0) {}

    ConstArray1D(const float* data, int32_t d1) : data_(data), d1_(d1) {}

    virtual ~ConstArray1D();

    // Init the object, the object does not own the data nor copy.
    // It is designed to directly use data from memory mapped resources.
    void init(const float* data, int32_t d1) {
        data_ = data;
        d1_ = d1;
    }

    // ReadArray1D methods.
    virtual int32_t d1() const override { return d1_; }
    virtual float get(int32_t i) const override {
        U_ASSERT(i < d1_);
        return data_[i];
    }

private:
    const float* data_;
    int32_t d1_;
};

ConstArray1D::~ConstArray1D()
{
}

/**
 * A class to index a float array as a 2D Array without owning the pointer or
 * copy the data.
 */
class ConstArray2D : public ReadArray2D {
public:
    ConstArray2D() : data_(nullptr), d1_(0), d2_(0) {}

    ConstArray2D(const float* data, int32_t d1, int32_t d2)
        : data_(data), d1_(d1), d2_(d2) {}

    virtual ~ConstArray2D();

    // Init the object, the object does not own the data nor copy.
    // It is designed to directly use data from memory mapped resources.
    void init(const float* data, int32_t d1, int32_t d2) {
        data_ = data;
        d1_ = d1;
        d2_ = d2;
    }

    // ReadArray2D methods.
    inline int32_t d1() const override { return d1_; }
    inline int32_t d2() const override { return d2_; }
    float get(int32_t i, int32_t j) const override {
        U_ASSERT(i < d1_);
        U_ASSERT(j < d2_);
        return data_[i * d2_ + j];
    }

    // Expose the ith row as a ConstArray1D
    inline ConstArray1D row(int32_t i) const {
        U_ASSERT(i < d1_);
        return ConstArray1D(data_ + i * d2_, d2_);
    }

private:
    const float* data_;
    int32_t d1_;
    int32_t d2_;
};

ConstArray2D::~ConstArray2D()
{
}

/**
 * A class to allocate data as a writable 1D array.
 * This is the main class implement matrix operation.
 */
class Array1D : public ReadArray1D {
public:
    Array1D() : memory_(nullptr), data_(nullptr), d1_(0) {}
    Array1D(int32_t d1)
        : memory_(calloc(d1, sizeof(float))),
          data_((float*)memory_), d1_(d1) {
    }

    virtual ~Array1D();

    // A special constructor which does not own the memory but writeable
    // as a slice of an array.
    Array1D(float* data, int32_t d1)
        : memory_(nullptr), data_(data), d1_(d1) {}

    // ReadArray1D methods.
    virtual int32_t d1() const override { return d1_; }
    virtual float get(int32_t i) const override {
        U_ASSERT(i < d1_);
        return data_[i];
    }

    // Return the index which point to the max data in the array.
    inline int32_t maxIndex() const {
        int32_t index = 0;
        float max = data_[0];
        for (int32_t i = 1; i < d1_; i++) {
            if (data_[i] > max) {
                max = data_[i];
                index = i;
            }
        }
        return index;
    }

    // Slice part of the array to a new one.
    inline Array1D slice(int32_t from, int32_t size) const {
        U_ASSERT(from >= 0);
        U_ASSERT(from < d1_);
        U_ASSERT(from + size <= d1_);
        return Array1D(data_ + from, size);
    }

    // Add dot product of a 1D array and a 2D array into this one.
    inline Array1D& addDotProduct(const ReadArray1D& a, const ReadArray2D& b) {
        U_ASSERT(a.d1() == b.d1());
        U_ASSERT(b.d2() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            for (int32_t j = 0; j < a.d1(); j++) {
                data_[i] += a.get(j) * b.get(j, i);
            }
        }
        return *this;
    }

    // Hadamard Product the values of another array of the same size into this one.
    inline Array1D& hadamardProduct(const ReadArray1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] *= a.get(i);
        }
        return *this;
    }

    // Add the Hadamard Product of two arrays of the same size into this one.
    inline Array1D& addHadamardProduct(const ReadArray1D& a, const ReadArray1D& b) {
        U_ASSERT(a.d1() == d1());
        U_ASSERT(b.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] += a.get(i) * b.get(i);
        }
        return *this;
    }

    // Add the values of another array of the same size into this one.
    inline Array1D& add(const ReadArray1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] += a.get(i);
        }
        return *this;
    }

    // Assign the values of another array of the same size into this one.
    inline Array1D& assign(const ReadArray1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] = a.get(i);
        }
        return *this;
    }

    // Apply tanh to all the elements in the array.
    inline Array1D& tanh() {
        return tanh(*this);
    }

    // Apply tanh of a and store into this array.
    inline Array1D& tanh(const Array1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1_; i++) {
            data_[i] = std::tanh(a.get(i));
        }
        return *this;
    }

    // Apply sigmoid to all the elements in the array.
    inline Array1D& sigmoid() {
        for (int32_t i = 0; i < d1_; i++) {
            data_[i] = 1.0f/(1.0f + expf(-data_[i]));
        }
        return *this;
    }

    inline Array1D& clear() {
        memset(data_, 0, d1_ * sizeof(float));
        return *this;
    }

private:
    void* memory_;
    float* data_;
    int32_t d1_;
};

Array1D::~Array1D()
{
    if (memory_) {
        free(memory_);
    }
}

class Array2D : public ReadArray2D {
public:
    Array2D() : memory_(nullptr), data_(nullptr), d1_(0), d2_(0) {}
    Array2D(int32_t d1, int32_t d2)
        : memory_(calloc(d1 * d2, sizeof(float))),
          data_((float*)memory_), d1_(d1), d2_(d2) {
    }
    virtual ~Array2D();

    // ReadArray2D methods.
    virtual int32_t d1() const override { return d1_; }
    virtual int32_t d2() const override { return d2_; }
    virtual float get(int32_t i, int32_t j) const override {
        U_ASSERT(i < d1_);
        U_ASSERT(j < d2_);
        return data_[i * d2_ + j];
    }

    inline Array1D row(int32_t i) const {
        U_ASSERT(i < d1_);
        return Array1D(data_ + i * d2_, d2_);
    }

    inline Array2D& clear() {
        memset(data_, 0, d1_ * d2_ * sizeof(float));
        return *this;
    }

private:
    void* memory_;
    float* data_;
    int32_t d1_;
    int32_t d2_;
};

Array2D::~Array2D()
{
    if (memory_) {
        free(memory_);
    }
}

typedef enum {
    BEGIN,
    INSIDE,
    END,
    SINGLE
} LSTMClass;

typedef enum {
    UNKNOWN,
    CODE_POINTS,
    GRAPHEME_CLUSTER,
} EmbeddingType;

struct LSTMBreakEngine::LSTMData {
    LSTMData(const lstm_data& model);
    const lstm_data& model;
    ConstArray2D fEmbedding;
    ConstArray2D fForwardW;
    ConstArray2D fForwardU;
    ConstArray1D fForwardB;
    ConstArray2D fBackwardW;
    ConstArray2D fBackwardU;
    ConstArray1D fBackwardB;
    ConstArray2D fOutputW;
    ConstArray1D fOutputB;
};

LSTMBreakEngine::LSTMData::LSTMData(const lstm_data& model): model(model) {
    int32_t mat1_size = (model.num_index + 1) * model.embedding_size;
    int32_t mat2_size = model.embedding_size * 4 * model.hunits;
    int32_t mat3_size = model.hunits * 4 * model.hunits;
    int32_t mat4_size = 4 * model.hunits;
    int32_t mat5_size = mat2_size;
    int32_t mat6_size = mat3_size;
    int32_t mat7_size = mat4_size;
    int32_t mat8_size = 2 * model.hunits * 4;
#if U_DEBUG
    int32_t mat9_size = 4;
    U_ASSERT(data_len == mat1_size + mat2_size + mat3_size + mat4_size + mat5_size +
        mat6_size + mat7_size + mat8_size + mat9_size);
#endif
    float *matrices = model.matrices;

    fEmbedding.init(matrices, (model.num_index + 1), model.embedding_size);
    matrices += mat1_size;
    fForwardW.init(matrices, model.embedding_size, 4 * model.hunits);
    matrices += mat2_size;
    fForwardU.init(matrices, model.hunits, 4 * model.hunits);
    matrices += mat3_size;
    fForwardB.init(matrices, 4 * model.hunits);
    matrices += mat4_size;
    fBackwardW.init(matrices, model.embedding_size, 4 * model.hunits);
    matrices += mat5_size;
    fBackwardU.init(matrices, model.hunits, 4 * model.hunits);
    matrices += mat6_size;
    fBackwardB.init(matrices, 4 * model.hunits);
    matrices += mat7_size;
    fOutputW.init(matrices, 2 * model.hunits, 4);
    matrices += mat8_size;
    fOutputB.init(matrices, 4);
}

// Computing LSTM as stated in
// https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
// ifco is temp array allocate outside which does not need to be
// input/output value but could avoid unnecessary memory alloc/free if passing
// in.
static void compute(
    int32_t hunits,
    const ReadArray2D& W, const ReadArray2D& U, const ReadArray1D& b,
    const ReadArray1D& x, Array1D& h, Array1D& c,
    Array1D& ifco)
{
    // ifco = x * W + h * U + b
    ifco.assign(b)
        .addDotProduct(x, W)
        .addDotProduct(h, U);

    ifco.slice(0*hunits, hunits).sigmoid();  // i: sigmod
    ifco.slice(1*hunits, hunits).sigmoid(); // f: sigmoid
    ifco.slice(2*hunits, hunits).tanh(); // c_: tanh
    ifco.slice(3*hunits, hunits).sigmoid(); // o: sigmod

    c.hadamardProduct(ifco.slice(hunits, hunits))
        .addHadamardProduct(ifco.slice(0, hunits), ifco.slice(2*hunits, hunits));

    h.tanh(c)
        .hadamardProduct(ifco.slice(3*hunits, hunits));
}

// Minimum word size
static const int32_t MIN_WORD = 2;

// Minimum number of characters for two words
static const int32_t MIN_WORD_SPAN = MIN_WORD * 2;

int32_t
LSTMBreakEngine::breakWord( const char32_t *text,
                            int32_t startPos,
                            int32_t endPos,
                            FoundBreakCallback foundBreak,
                            void* callbackContext) const {
    int32_t input_seq_len = endPos - startPos;
    if (input_seq_len > 2048) {
        // give up breaking this sentence rather than risk going out-of-memory
        return -1;
    }

    int32_t* indices = (int32_t*) malloc(input_seq_len * sizeof(int32_t));
    for (int i = 0; i < input_seq_len; i++) {
        indices[i] = fData->model.mapping(text[startPos + i]);
#ifdef LSTM_VECTORIZER_DEBUG
        printf("[U+%04x ] map to %d\n", text[startPos + i], indices[i]);
#endif
    }

    int32_t hunits = fData->fForwardU.d1();

    // ----- Begin of all the Array memory allocation needed for this function
    // Allocate temp array used inside compute()
    Array1D ifco(4 * hunits);

    Array1D c(hunits);
    Array1D logp(4);

    // TODO: limit size of hBackward. If input_seq_len is too big, we could
    // run out of memory.
    // Backward LSTM
    Array2D hBackward(input_seq_len, hunits);

    // Allocate fbRow and slice the internal array in two.
    Array1D fbRow(2 * hunits);

    // To save the needed memory usage, the following is different from the
    // Python or ICU4X implementation. We first perform the Backward LSTM
    // and then merge the iteration of the forward LSTM and the output layer
    // together because we only neetdto remember the h[t-1] for Forward LSTM.
    for (int32_t i = input_seq_len - 1; i >= 0; i--) {
        Array1D hRow = hBackward.row(i);
        if (i != input_seq_len - 1) {
            hRow.assign(hBackward.row(i+1));
        }
#ifdef LSTM_DEBUG
        printf("hRow %d\n", i);
        hRow.print();
        printf("indicesBuf[%d] = %d\n", i, indices[i]);
        printf("fData->fEmbedding.row(indicesBuf[%d]):\n", i);
        fData->fEmbedding.row(indices[i]).print();
#endif  // LSTM_DEBUG
        compute(hunits,
                fData->fBackwardW, fData->fBackwardU, fData->fBackwardB,
                fData->fEmbedding.row(indices[i]),
                hRow, c, ifco);
    }


    Array1D forwardRow = fbRow.slice(0, hunits);  // point to first half of data in fbRow.
    Array1D backwardRow = fbRow.slice(hunits, hunits);  // point to second half of data n fbRow.

    // The following iteration merge the forward LSTM and the output layer
    // together.
    c.clear();  // reuse c since it is the same size.
    for (int32_t i = 0; i < input_seq_len; i++) {
#ifdef LSTM_DEBUG
        printf("forwardRow %d\n", i);
        forwardRow.print();
#endif  // LSTM_DEBUG
        // Forward LSTM
        // Calculate the result into forwardRow, which point to the data in the first half
        // of fbRow.
        compute(hunits,
                fData->fForwardW, fData->fForwardU, fData->fForwardB,
                fData->fEmbedding.row(indices[i]),
                forwardRow, c, ifco);

        // assign the data from hBackward.row(i) to second half of fbRowa.
        backwardRow.assign(hBackward.row(i));

        logp.assign(fData->fOutputB).addDotProduct(fbRow, fData->fOutputW);
#ifdef LSTM_DEBUG
        printf("backwardRow %d\n", i);
        backwardRow.print();
        printf("logp %d\n", i);
        logp.print();
#endif  // LSTM_DEBUG

        // current = argmax(logp)
        LSTMClass current = (LSTMClass)logp.maxIndex();
        // BIES logic.
        if (current == BEGIN || current == SINGLE) {
            if (i != 0) {
                (*foundBreak)(callbackContext, startPos + i);
            }
        }
    }

    free(indices);
    return 0;
}

LSTMBreakEngine::LSTMBreakEngine(const lstm_data& model)
{
    fData = new LSTMData(model);
}

LSTMBreakEngine::~LSTMBreakEngine() {
    delete fData;
}
