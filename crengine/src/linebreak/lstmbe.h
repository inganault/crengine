// © 2021 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#ifndef LSTMBE_H
#define LSTMBE_H

#include "lstm_data.h"

class LSTMBreakEngine {
public:
    LSTMBreakEngine(const lstm_data& model);
    ~LSTMBreakEngine();

    typedef void (*FoundBreakCallback)(void* context, int32_t pos);

    /**
     * <p>Divide up a range of known dictionary characters handled by this break engine.</p>
     *
     * @param text A UText representing the text
     * @param rangeStart The start of the range of dictionary characters
     * @param rangeEnd The end of the range of dictionary characters
     * @param foundBreak Callback to call when found a break
     * @param callbackContext Argument when calling foundBreak
     * @return non-zero if error
     */
     int32_t breakWord( const char32_t *text,
                        int32_t rangeStart,
                        int32_t rangeEnd,
                        FoundBreakCallback foundBreak,
                        void* callbackContext) const;
private:
    struct LSTMData;
    const LSTMData* fData;
};

#endif  /* LSTMBE_H */
