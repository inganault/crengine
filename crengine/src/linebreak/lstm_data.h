#ifndef LSTM_DATA_H
#define LSTM_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t (*lstm_data_mapping)(int32_t codepoint);

struct lstm_data {
    int num_index;
    int embedding_size;
    int hunits;
    lstm_data_mapping mapping;
    float *matrices;
};

extern struct lstm_data lstm_model_thai;
extern struct lstm_data lstm_model_lao;
extern struct lstm_data lstm_model_burmese;
extern struct lstm_data lstm_model_khmer;

#ifdef __cplusplus
}
#endif

#endif /* LSTM_DATA_H */
