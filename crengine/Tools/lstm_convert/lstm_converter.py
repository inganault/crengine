#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Convert lstm model data from ICU to C code
# depends only on python standard libraries

import json
import struct
from inspect import cleandoc
from pathlib import Path
import sys

OUTPUT_PATH = '../../src/linebreak/lstm_data.c'

LSTM_DATA_TYPE = 'struct lstm_data'
EXPORT_PREFIX = 'lstm_model_'

INTERNAL_PREFIX = 'lstm_data_'

# Models directory path, can be found in repo below
LSTM_WORD_SEGMENTATION_PATH = './models'

# Url to download models if not found
MODEL_URL = 'https://github.com/unicode-org/lstm_word_segmentation/releases/download/v0.1.0/models.zip'

models = [
    ('thai', LSTM_WORD_SEGMENTATION_PATH + '/Thai_codepoints_exclusive_model4_heavy/weights.json'),
    ('lao', LSTM_WORD_SEGMENTATION_PATH + '/Lao_codepoints_exclusive_model4_heavy/weights.json'),
    ('burmese', LSTM_WORD_SEGMENTATION_PATH + '/Burmese_codepoints_exclusive_model4_heavy/weights.json'),
    ('khmer', LSTM_WORD_SEGMENTATION_PATH + '/Khmer_codepoints_exclusive_model4_heavy/weights.json'),
]

# Convert dictionary to continuous ranges
def coalesce_dict(dic):
    kv = sorted(dic.items())
    key_start = -2
    value_start = -2
    block_length = 0
    ranges = []
    for k, v in kv:
        k = ord(k)
        if (key_start + block_length == k and
           value_start + block_length == v):
            # continuous
            block_length += 1
        else:
            if block_length != 0:
                ranges += [(key_start, value_start, block_length)]
            key_start, value_start = k, v
            block_length = 1
    if block_length != 0:
        ranges += [(key_start, value_start, block_length)]
    return ranges

# Convert ranges to C code
def ranges_to_code(fn_name, ranges):
    code = f'static int32_t {fn_name}(int32_t codepoint) {{\n'
    for key_start, value_start, block_length in ranges:
        if block_length == 1:
            code += f'    if (0x{key_start:04x} == codepoint) {{\n'
            code += f'        return {value_start};\n'
            code +=  '    }\n'
        else:
            key_end = key_start + block_length
            code += f'    if (0x{key_start:04x} <= codepoint && codepoint < 0x{key_end:04x}) {{\n'
            code += f'        return codepoint - 0x{key_start:04x} + {value_start};\n'
            code +=  '    }\n'
    code += '    return -1;\n'
    code += '}\n'
    return code

to_float32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]

def matrix_to_struct(name, array):
    code = f'static float {name}[] = {{\n    '
    for val in array:
        code += f'{val:.9g}f, '
        # verify that float have enough precision
        assert to_float32(float(f'{val:.9g}')) == to_float32(val)
    code = code[:-2]
    code += '\n};\n'
    return code

# print to file instead of stdout
output = open(OUTPUT_PATH, 'w', encoding='utf-8')
def gencode(*args, **kwargs):
    return print(*args, **kwargs, file=output)
def stderr(*args, **kwargs):
    return print(*args, **kwargs, file=sys.stderr)

# Auto download

if not Path(LSTM_WORD_SEGMENTATION_PATH).exists():
    import urllib.request
    import zipfile
    stderr("Models not found, auto downloading ...")
    mdl_path = Path(LSTM_WORD_SEGMENTATION_PATH)
    mdl_path.mkdir()
    zip_path = mdl_path / "models.zip"
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(mdl_path)
    zip_path.unlink()
    stderr("Download & extract completed")

# Code generation starts here

gencode('''
// AUTO-GENRATED CODE FROM lstm_converter.py
// Model data for LSTM taken from International Components for Unicode (ICU)
// License & terms of use: http://www.unicode.org/copyright.html

#include "lstm_data.h"

'''.lstrip(), end='')
for name, weight_path in models:
    with open(weight_path, 'rb') as weight:
        model = json.load(weight)

    num_index = len(model['dic'])
    embedding_size = model['mat1']['dim'][1]
    hunits = model['mat3']['dim'][0]


    gencode(f'// --- {name.title()} ---')
    gencode()
    prefix = f'{INTERNAL_PREFIX}{name}_'

    # print mapping function
    gencode(ranges_to_code(f'{prefix}dict', coalesce_dict(model['dic'])))

    # print weights (matrix)
    all_matrix = []
    for mat_id in range(1,9+1):
        mat_name = f'mat{mat_id}'
        all_matrix += model[mat_name]['data']
    gencode(matrix_to_struct(f'{prefix}mat', all_matrix))

    gencode(cleandoc(f'''
        {LSTM_DATA_TYPE} {EXPORT_PREFIX}{name} = {{
            {num_index}, // num_index
            {embedding_size}, // embedding_size
            {hunits}, // hunits
            &{prefix}dict, // mapping function
            {prefix}mat // matrices
        }};
    '''))
    gencode()

output.close()
stderr('Finished.')
