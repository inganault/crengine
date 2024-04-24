#ifndef LINEBREAK_SA_H
#define LINEBREAK_SA_H

#include "lvtypes.h"

typedef void (*FoundBreakCallback)(void* context, int32_t pos);

/**
 * Break Complex context dependent (South East Asian) characters into words
 *
 * @param text A lChar32 representing the text
 * @param rangeStart The start of the range
 * @param rangeEnd The end of the range
 * @param foundBreak Callback to call when found a break
 * @param callbackContext Argument when calling foundBreak
 * @return non-zero if error
 */
int32_t BreakSALine( const lChar32 *text,
                   int32_t rangeStart,
                   int32_t rangeEnd,
                   FoundBreakCallback foundBreak,
                   void* callbackContext);

#endif /* LINEBREAK_SA_H */
