# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#define your pre-trained (post-trained) models here with their paths.

MODEL_ARCHIVE_MAP = {
    'laptop_pt': 'pt_model/laptop_pt/',
    'rest_pt': 'pt_model/rest_pt/',
    'bert_base': 'bert-base-cased',
    'roberta_base': 'roberta-base',
    'phobert_base': 'vinai/phobert-base',
    'laptop_phobert_pt': 'pt_model/laptop_phobert_pt'
}
