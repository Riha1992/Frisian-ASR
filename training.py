from huggingface_hub import login
login(token="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")

model_number = "40"

model_name = "wav2vec2_fy_nl_en_common_voice_"

repo_name=model_name+model_number

seed=34253
from transformers import set_seed
set_seed(seed)


# load the dataset
import datasets
from datasets import load_dataset, load_metric



base_url_train = "/data/p312702/from_wietse/train/"
base_url_validation="/data/p312702/from_wietse/validation/"
#base_url_test="/data/p312702/from_wietse/test/"

cv_germanic_train = load_dataset("audiofolder",data_dir=base_url_train)
cv_germanic_validation=load_dataset("audiofolder",data_dir=base_url_validation)
#cv_germanic_test=load_dataset("audiofolder",data_dir=base_url_test)

#print(cv_germanic_train)


def filter_function(example):
    # Example: filter rows where a specific column equals a certain value
    return example['lid'] in ['fy-NL','nl','en']



cv_frisian_nl_de_train = cv_germanic_train.filter(filter_function)
cv_frisian_nl_de_validation = cv_germanic_validation.filter(filter_function)
#cv_frisian_nl_de_test=cv_germanic_test.filter(filter_function)

#print(cv_frisian_train)
#print(cv_frisian_validation)




# show random elements
from datasets import ClassLabel
import random
import pandas as pd
import torchaudio
import re

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    df = pd.DataFrame(dataset[picks])
    #print(df)
    #display(HTML(df.to_html()))


# remove special characters (normalization)
import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', str(batch["transcription"])).lower()
    #batch["transcription"] = "["+(batch["lid"].upper())+"]"+" "+batch["transcription"]
    return batch

cv_frisian_nl_de_train = cv_frisian_nl_de_train.map(remove_special_characters)
cv_frisian_nl_de_validation = cv_frisian_nl_de_validation.map(remove_special_characters)
#cv_frisian_nl_de_test=cv_frisian_nl_de_test.map(remove_special_characters)

#cv_frisian_nl_de_train=cv_frisian_nl_de_train.map(remove_special_characters)
#cv_frisian_nl_de_validation=cv_frisian_nl_de_validation.map(remove_special_characters)
#cv_frisian_nl_de_test=cv_frisian_nl_de_test.map(remove_special_characters)

#show_random_elements(cv_germanic_train["train"].remove_columns(["audio"]))


def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}






#print(cv_germanic_train)
#print(cv_germanic_validation)
#print(cv_germanic_train.column_names['train'][0])
vocabs_train = cv_frisian_nl_de_train.map(extract_all_chars) # , batched=True, batch_size=-1, keep_in_memory=True,remove_columns=cv_germanic_train.column_names[0])
vocabs_validation=cv_frisian_nl_de_validation.map(extract_all_chars)


vocabs_train = vocabs_train.remove_columns(["transcription","lid","audio"])
vocabs_validation = vocabs_validation.remove_columns(["transcription","lid","audio"])


#print(vocabs_train["train"]["vocab"])

merged_list_train = []
merged_list_validation = []

for sublist in vocabs_train["train"]["vocab"]:
    merged_list_train.extend(sublist)

for sublist in vocabs_validation["train"]["vocab"]:
     merged_list_validation.extend(sublist)

#print(merged_list_train)

merged_list_train=[element for sublist in merged_list_train for element in sublist]
merged_list_validation=[element for sublist in merged_list_validation for element in sublist]

#print(merged_list_train)
#print(merged_list_validation)

vocab_list = list(set(merged_list_train) | set(merged_list_validation))


vocab_dict = {v: k for k, v in enumerate(vocab_list)}


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
#vocab_dict["[EN]"] = len(vocab_dict)
#vocab_dict["[NL]"] = len(vocab_dict)
#vocab_dict["[DE]"] = len(vocab_dict)
#vocab_dict["[FY-NL]"] = len(vocab_dict)
print(len(vocab_dict))





#print(len(vocab_dict))

# export vocab json
import json
with open('/data/p312702/from_wietse/'+model_name+model_number+'/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)






from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("/data/p312702/from_wietse/"+model_name+model_number+"/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
repo_name = model_name+model_number
tokenizer.push_to_hub(repo_name)

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)#48000
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)




def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch



# This may take a long while
# save to disk

cv_frisian_nl_de_train=cv_frisian_nl_de_train.map(prepare_dataset)
cv_frisian_nl_de_validation=cv_frisian_nl_de_validation.map(prepare_dataset)
#cv_frisian_nl_de_test=cv_frisian_nl_de_test.map(prepare_dataset)


cv_frisian_nl_de_train=cv_frisian_nl_de_train.remove_columns(["transcription","audio","lid"])
cv_frisian_nl_de_validation=cv_frisian_nl_de_validation.remove_columns(["transcription","audio","lid"])
#cv_frisian_nl_de_test=cv_frisian_nl_de_test.remove_columns(["transcription","audio","lid"])



# save to disk:
#from datasets import load_from_disk
#from transformers import save_to_disk


#prepared_data_directory_train = "/data/p312702/from_wietse/prepared_data/train"
#prepared_data_directory_validation = "/data/p312702/from_wietse/prepared_data/validation"


#cv_germanic_train.save_to_disk(prepared_data_directory_train)
#cv_germanic_validation.save_to_disk(prepared_data_directory_validation)



#from datasets import load_from_disk
# load from disk: 
#cv_germanic_train_prepared=load_from_disk(prepared_data_directory_train)
#cv_germanic_validation_prepared=load_from_disk(prepared_data_directory_validation)


#print(cv_germanic_train_prepared)
#print("*****************************")
#print(cv_germanic_validation_prepared)


#cv_germanic_train=cv_germanic_train_prepared
#cv_germanic_validation=cv_germanic_validation_prepared




import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)



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

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-1b",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    ignore_mismatched_sizes=True,
    vocab_size=150, #64,
)

model.freeze_feature_encoder()


from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  seed=seed,
  data_seed=seed,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  gradient_accumulation_steps=16,
  eval_accumulation_steps=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=100,
  save_strategy="no",
  eval_steps=100,
  logging_steps=100,
  learning_rate=8e-5,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
  adam_beta2=0.98,
  warmup_ratio=0.1,
)


# Here you can choose whether to use validated data only or use validated+invalidated
#common_voice_train_invalidated = datasets.concatenate_datasets([common_voice["train"],common_voice["invalidated"]])
#train_datasett = common_voice_train_invalidated 
train_dataset = cv_frisian_nl_de_train["train"]
eval_dataset = cv_frisian_nl_de_validation["train"]
#test_dataset = cv_germanic_test



#print(train_datasett["transcription"])

print("****************************************************************")
print("******************************start training*********************")
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset, # common_voice["train]
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)


trainer.train()

trainer.save_model(repo_name)
'''
# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder( 
    #use the correct checkpoint noting whether only validated data is used. 
    folder_path="/nvme/odds-rug/reihaneh/wav2vec2_germanic_common_voice_"+model_number+"/checkpoint-50",
    repo_id="Reihaneh/"+repo_name,
    repo_type="model",
)





from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("Reihaneh/wav2vec2_germanic_common_voice_"+model_number)
model = Wav2Vec2ForCTC.from_pretrained("Reihaneh/wav2vec2_germanic_common_voice_"+model_number).cuda()



def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  # batch["pred_str"] = processor.batch_decode(pred_ids) #[0][0] 
  batch["transcription"] = processor.decode(batch["labels"], group_tokens=False)  
  return batch

results = cv_frisian_nl_test.map(map_to_result)


#print(results)
results = results["train"]

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["transcription"])))

#___________________________________________________________________________________________________
#show_random_elements(results.remove_columns(["speech", "sampling_rate"]))


'''
