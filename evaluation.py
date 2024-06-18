from huggingface_hub import login
login(token="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")

from datasets import Audio

#model_number = "36"


#target_checkpoint="30100"
#"wav2vec2_fy_nl_with_lid_22_gas16_bs8_emax30"
#model_id="wav2vec2_fy_nl_en_with_lid_20_gas16_bs8_emax30"
#model_id="wav2vec2_fy_nl_en_de_common_voice_16_gas16_bs8_emax30"
#model_id="wav2vec2_fy_nl_with_lid_22_gas16_bs8_emax30"
#model_id = "wav2vec2_fy_nl_common_voice_17_gas16_bs8_emax30_old"
model_id="wav2vec2_fy_nl_en_common_voice_40"
#model_id = "wav2vec2_fy_common_voice_30_gas16_bs8_emax50"
#model_name = "wav2vec2_fy_common_voice_"
#model_id=model_name+model_number



# /data/p312702/wav2vec2_frisian_common_voice_14
#__________________________________________________________________________________________________________
#seed=668912
#from transformers import set_seed
#set_seed(seed)
#________________________________________________________________________________________________________
# load the dataset
import datasets
from datasets import load_dataset, load_metric
#common_voice = load_dataset("fsicoli/common_voice_17_0", "fy-NL",use_auth_token=True, trust_remote_code=True )
#common_voice = common_voice.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))
#common_voice = common_voice.remove_columns(['client_id', 'up_votes','down_votes', 'age', 'gender','accent','locale','segment'])


base_url_test="/data/p312702/from_wietse/test/"
#base_url_test="/data/p312702/from_wietse/validation"

#base_url_test="/data/p312702/SPRAAKLAB/R_2_3_4_8_9_10_13_14_17_18_19_DUTCH"
#base_url_test="/data/p312702/SPRAAKLAB/R_1_5_6_7_11_12_15_16_20_FRYSK"


cv_germanic_test=load_dataset("audiofolder",data_dir=base_url_test,trust_remote_code=True)
cv_frisian_test=cv_germanic_test

def filter_function(example):
    # Example: filter rows where a specific column equals a certain value
    #return example['lid'] in ['fy-NL','nl','de']
    return example['lid'] == ('fy-NL')

cv_frisian_test=cv_frisian_test.filter(filter_function)


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(cv_frisian_test)

cv_frisian_test['train'] = cv_frisian_test['train'].cast_column("audio", Audio(sampling_rate=16000))

#______________________________________________________________________________________________________
# show random elements
from datasets import ClassLabel
import random
import pandas as pd
import torchaudio



import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', str(batch["transcription"])).lower()
    batch["transcription"] = batch["transcription"]
    return batch

#cv_frisian_nl_de_train = cv_frisian_nl_de_train.map(remove_special_characters)
#cv_frisian_nl_de_validation = cv_frisian_nl_de_validation.map(remove_special_characters)
cv_frisian_test=cv_frisian_test.map(remove_special_characters)


from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/data/p312702/from_wietse/"+model_id, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False) #48000
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

'''
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


# This may take a long while
cv_frisian_test = cv_frisian_test.map(prepare_dataset, remove_columns=cv_frisian_test.column_names["train"], num_proc=1)
'''
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_metric
wer_metric = load_metric("wer")


import numpy as np
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor



#feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-1b")

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#processor = Wav2Vec2Processor.from_pretrained("/data/p312702/from_wietse/wav2vec2_germanic_common_voice_3") #"Reihaneh/wav2vec2_frisian_common_voice_"+model_number)
model = Wav2Vec2ForCTC.from_pretrained("/data/p312702/from_wietse/"+model_id).cuda() #"/checkpoint-"+target_checkpoint).cuda() # "Reihaneh/wav2vec2_frisian_common_voice_"+model_number).cuda()


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


# This may take a long while
cv_frisian_test = cv_frisian_test.map(prepare_dataset, remove_columns=cv_frisian_test.column_names["train"], num_proc=1)

language_ids=[]

def map_to_result(batch):
  model.to("cuda") 
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  #print(pred_ids)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0].split("]")[-1]
  print(batch["pred_str"])
  batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
  print(batch["sentence"])
  return batch

wer_metric = load_metric("wer")
results = cv_frisian_test.map(map_to_result, remove_columns=cv_frisian_test['train'].column_names)
print(results)

#results = frisian_test.map(map_to_result, remove_columns=frisian_test.column_names)


print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["train"]["pred_str"], references=results["train"]["sentence"])))
#___________________________________________________________________________________________________
#show_random_elements(results.remove_columns(["speech", "sampling_rate"]))

