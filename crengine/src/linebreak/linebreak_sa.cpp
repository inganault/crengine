#include "../../include/linebreak_sa.h"
#include "lstmbe.h"
#include "lstm_data.h"

enum class SALang {
    THAI,
    LAO,
    BURMESE,
    KHMER,
    UNK
};
static SALang classify_language(const lChar32 ch){
    if (0x0E00 <= ch && ch < 0x0E80)
        return SALang::THAI;
    if (0x0E80 <= ch && ch < 0x0F00)
        return SALang::LAO;
    if (0x1000 <= ch && ch < 0x10A0)
        return SALang::BURMESE;
    if (0x1780 <= ch && ch < 0x1800)
        return SALang::KHMER;
    return SALang::UNK;
}

#define def_engine_singleton(fn_name, data) \
    static LSTMBreakEngine &fn_name() { \
        static LSTMBreakEngine engine(data); \
        return engine; \
    }

def_engine_singleton(get_break_engine_singleton_thai,    lstm_model_thai)
def_engine_singleton(get_break_engine_singleton_lao,     lstm_model_lao)
def_engine_singleton(get_break_engine_singleton_burmese, lstm_model_burmese)
def_engine_singleton(get_break_engine_singleton_khmer,   lstm_model_khmer)

static LSTMBreakEngine &get_break_engine_by_lang(SALang lang) {
    switch (lang) {
        case SALang::THAI:
            return get_break_engine_singleton_thai();
        case SALang::LAO:
            return get_break_engine_singleton_lao();
        case SALang::BURMESE:
            return get_break_engine_singleton_burmese();
        case SALang::KHMER:
            return get_break_engine_singleton_khmer();
    }
    return get_break_engine_singleton_thai(); // supress warning
}

int32_t BreakSALine( const lChar32 *text,
                   int32_t rangeStart,
                   int32_t rangeEnd,
                   FoundBreakCallback foundBreak,
                   void* callbackContext) {
    int lang_chunk_start = rangeStart;
    SALang chunk_lang = SALang::UNK;

    for(int pos = rangeStart; pos < rangeEnd; pos++) {
        SALang lang = classify_language(text[pos]);
        if(lang != chunk_lang) {
            if (chunk_lang != SALang::UNK) {
                auto &engine = get_break_engine_by_lang(chunk_lang);
                engine.breakWord(
                    (const char32_t *)text, lang_chunk_start, pos,
                    foundBreak, callbackContext
                );
            }
            chunk_lang = lang;
            lang_chunk_start = pos;
        }
    }
    if (lang_chunk_start != rangeEnd && chunk_lang != SALang::UNK) {
        auto &engine = get_break_engine_by_lang(chunk_lang);
        engine.breakWord(
            (const char32_t *)text, lang_chunk_start, rangeEnd,
            foundBreak, callbackContext
        );
    }
    return 0;
}
