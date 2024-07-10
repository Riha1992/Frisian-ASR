import soundfile as sf
from datasets import load_dataset
import csv
import datasets 
import os
import re


from huggingface_hub import login
login(token="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")



#n_max_train = 34605
#n_max_train = 3921
n_max_train = 3921
n_max_valid = 3171


#fsicoli/common_voice_17_0

#dataset_train_de = load_dataset('mozilla-foundation/common_voice_17_0','de', split='train', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_train_de = dataset_train_de.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_train_nl = load_dataset('mozilla-foundation/common_voice_17_0','nl', split='train', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_train_nl = dataset_train_nl.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_train_sv = load_dataset('mozilla-foundation/common_voice_17_0', 'sv-SE', split='train', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_train_sv = dataset_train_sv.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

dataset_train_fy = load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='train', streaming=True, trust_remote_code=True, use_auth_token=True)
dataset_train_fy = dataset_train_fy.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_train_en = load_dataset('mozilla-foundation/common_voice_17_0', 'en', split='train', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_train_en = dataset_train_en.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))





#dataset_test_en = load_dataset('mozilla-foundation/common_voice_17_0', 'en', split='test', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_test_en = dataset_test_en.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_test_de = load_dataset('mozilla-foundation/common_voice_17_0', 'de', split='test', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_test_de = dataset_test_de.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_test_nl= load_dataset('mozilla-foundation/common_voice_17_0', 'nl', split='test', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_test_nl = dataset_test_nl.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_test_sv = load_dataset('mozilla-foundation/common_voice_17_0', 'sv-SE', split='test', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_test_sv = dataset_test_sv.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

dataset_test_fy= load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='test', streaming=True, trust_remote_code=True, use_auth_token=True)
dataset_test_fy = dataset_test_fy.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))




#dataset_validation_en =load_dataset('mozilla-foundation/common_voice_17_0', 'en', split='validation', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_validation_en = dataset_validation_en.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_validation_de =load_dataset('mozilla-foundation/common_voice_17_0', 'de', split='validation', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_validation_de = dataset_validation_de.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_validation_nl =load_dataset('mozilla-foundation/common_voice_17_0', 'nl', split='validation', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_validation_nl = dataset_validation_nl.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_validation_sv =load_dataset('mozilla-foundation/common_voice_17_0', 'sv-SE', split='validation', streaming=True, trust_remote_code=True, use_auth_token=True)
#dataset_validation_sv = dataset_validation_sv.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

dataset_validation_fy =load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='validation', streaming=True, trust_remote_code=True, use_auth_token=True)
dataset_validation_fy = dataset_validation_fy.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))


#dataset_invalidated_en = load_dataset('mozilla-foundation/common_voice_16_1', 'en', split='invalidated', streaming=True)
#dataset_invalidated_de = load_dataset('mozilla-foundation/common_voice_16_1', 'de', split='invalidated', streaming=True)
#dataset_invalidated_nl =load_dataset('mozilla-foundation/common_voice_16_1', 'nl', split='invalidated', streaming=True)
#dataset_invalidated_sv_SE =load_dataset('mozilla-foundation/common_voice_16_1', 'sv-SE', split='invalidated', streaming=True)
#dataset_invalidated_fy =load_dataset('mozilla-foundation/common_voice_16_1', 'fy-NL', split='invalidated', streaming=True)



#dataset_other_en = load_dataset('mozilla-foundation/common_voice_16_1', 'en', split='other', streaming=True)
#dataset_other_de = load_dataset('mozilla-foundation/common_voice_16_1', 'de', split='other', streaming=True)
#dataset_other_nl= load_dataset('mozilla-foundation/common_voice_16_1', 'nl', split='other', streaming=True)
#datase_other_sv_SE =load_dataset('mozilla-foundation/common_voice_16_1', 'sv-SE', split='other', streaming=True)
#dataset_other_fy=load_dataset('mozilla-foundation/common_voice_16_1', 'fy-NL', split='other', streaming=True)


# list_of_rows_train_nl = list(dataset_train_nl)    # 34605                 # all: 36.4k
# list_of_rows_test_nl = list(dataset_test_nl)             # 11235          # all:11.2k
# list_of_rows_validation_nl = list(dataset_validation_nl)    # 11223       # all:11.2k
# #list_of_rows_invalidated_nl = list(dataset_invalidated_nl.take(5540))            # all: 5.54k
# #list_of_rows_other_nl=list(dataset_other_nl.take(2290))                          # all: 2.29k
# print("nl", len(list_of_rows_train_nl), len(list_of_rows_test_nl), len(list_of_rows_validation_nl))

# list_of_rows_train_en = list(dataset_train_en.take(len(list_of_rows_train_nl)))   #34605            # all: 114k
# list_of_rows_test_en = list(dataset_test_en)           #11235      # all: 16.4k
# list_of_rows_validation_en = list(dataset_validation_en)  # 11223  # all: 16.4k
# #list_of_rows_invalidated_en = list(dataset_invalidated_en.take(5800))     # all: 111k
# #list_of_rows_other_en = list(dataset_other_en.take(5800))                 # all: 154k
# print("en", len(list_of_rows_train_en), len(list_of_rows_test_en), len(list_of_rows_validation_en))


# list_of_rows_train_de = list(dataset_train_de.take(len(list_of_rows_train_nl)))    #34605            # all : 121k
# list_of_rows_test_de = list(dataset_test_de)           #11235       # all:  16.2k
# list_of_rows_validation_de = list(dataset_validation_de) # 11223    # all: 16.2k
# #list_of_rows_invalidated_de = list(dataset_invalidated_de.take(5800))      # all: 53.8k
# #list_of_rows_other_de=list(dataset_other_de.take(5800))                    # all: 7k
# print("de", len(list_of_rows_train_de), len(list_of_rows_test_de), len(list_of_rows_validation_de))


# list_of_rows_train_sv_SE = list(dataset_train_sv_SE) # 7657                # all: 7.66k
# list_of_rows_test_sv_SE = list(dataset_test_sv_SE)          #5206          # all: 5.21k
# list_of_rows_validation_sv_SE = list(dataset_validation_sv_SE)  # 5222     # all: 5.22k
# #list_of_rows_invalidated_sv_SE = list(dataset_invalidated_sv_SE.take(1420))       # all: 1.42k
# #list_of_rows_other_sv_SE=list(dataset_other_sv_SE.take(6610))                     # all: 6.61k
# print("sv-SE", len(list_of_rows_train_sv_SE), len(list_of_rows_test_sv_SE), len(list_of_rows_validation_sv_SE))


# list_of_rows_train_fy = list(dataset_train_fy)   #3920                 # all: 3.92k
# list_of_rows_test_fy=list(dataset_test_fy)       # 3171                # all: 3.17k
# list_of_rows_validation_fy = list(dataset_validation_fy) # 3171        # all:3.17k
# #list_of_rows_invalidated_fy = list(dataset_invalidated_fy.take(3950))         # all: 3.95k
# #list_of_rows_other_fy=list(dataset_other_fy.take(102000))                     # all: 102k
# print("fy-NL", len(list_of_rows_train_fy), len(list_of_rows_test_fy), len(list_of_rows_validation_fy))




#print(list_of_rows_train_de)
whole_rows_train = []
multilingual_dataset_list_train = []
#multilingual_dataset_list_train.append([dataset_train_en,"en"])
#multilingual_dataset_list_train.append([dataset_train_de,"de"])
#multilingual_dataset_list_train.append([dataset_train_nl,"nl"])
#multilingual_dataset_list_train.append([dataset_train_sv,"sv-SE"])
multilingual_dataset_list_train.append([dataset_train_fy,"fy-NL"])


whole_rows_test = []
multilingual_dataset_list_test = []
#multilingual_dataset_list_test.append([dataset_test_en,"en"])
#multilingual_dataset_list_test.append([dataset_test_de,"de"])
#multilingual_dataset_list_test.append([dataset_test_nl,"nl"])
#multilingual_dataset_list_test.append([dataset_test_sv,"sv-SE"])
multilingual_dataset_list_test.append([dataset_test_fy,"fy-NL"])


whole_rows_validation = []
multilingual_dataset_list_validation = []
#multilingual_dataset_list_validation.append([dataset_validation_en,"en"])
#multilingual_dataset_list_validation.append([dataset_validation_de,"de"])
#multilingual_dataset_list_validation.append([dataset_validation_nl,"nl"])
#multilingual_dataset_list_validation.append([dataset_validation_sv,"sv-SE"])
multilingual_dataset_list_validation.append([dataset_validation_fy,"fy-NL"])


whole_rows_test_fy_NL=[]
#whole_rows_test_sv_SE=[]
#whole_rows_test_en=[]
#whole_rows_test_de=[]
#whole_rows_test_nl=[]


#clips_directory = "clips/"
os.makedirs("/data/p312702/s3_router/train/", exist_ok=True)
os.makedirs("/data/p312702/s3_router/test/", exist_ok=True)
os.makedirs("/data/p312702/s3_router/validation/",exist_ok=True)


print("train")
for pair in multilingual_dataset_list_train:
	LID=pair[1]
	print(LID)
	n = 0
	for row_dict in iter(pair[0]):
		audio_content=row_dict['audio']['array']
		filename = row_dict['audio']['path'].split("/")[1].replace(".mp3",".wav")
		sentence_without_comma = re.sub(r',', ' ', row_dict['sentence'])
		whole_rows_train.append([filename, sentence_without_comma, LID]) #up_votes, down_votes, age, gender, accent, locale, segment, variant
		sf.write("train/"+filename, audio_content, 16000)
		n += 1
		if n == n_max_train:
			break

print("test")
for pair in multilingual_dataset_list_test:
	LID=pair[1]
	print(LID)
	for row_dict in iter(pair[0]):
		audio_content=row_dict['audio']['array']
		filename = row_dict['audio']['path'].split("/")[1].replace(".mp3",".wav")
		sentence_without_comma = re.sub(r',', ' ', row_dict['sentence'])
		whole_rows_test.append([filename, sentence_without_comma, LID]) #up_votes, down_votes, age, gender, accent, locale, segment, variant
		sf.write("test/"+filename, audio_content, 16000)
		if LID=="fy-NL":
			whole_rows_test_fy_NL.append([filename, sentence_without_comma, LID])
		elif LID=="sv-SE":
			whole_rows_test_sv_SE.append([filename, sentence_without_comma, LID])
		elif LID=="nl":
			whole_rows_test_nl.append([filename, sentence_without_comma, LID])
		elif LID=="de":
			whole_rows_test_de.append([filename, sentence_without_comma, LID])
		elif LID=="en":
			whole_rows_test_en.append([filename, sentence_without_comma, LID])

print("validation")
for pair in multilingual_dataset_list_validation:
	LID=pair[1]
	print(LID)
	n = 0
	for row_dict in iter(pair[0]):
		audio_content=row_dict['audio']['array']
		sentence_without_comma = re.sub(r',', ' ', row_dict['sentence'])
		filename = row_dict['audio']['path'].split("/")[1].replace(".mp3",".wav")
		whole_rows_validation.append([filename, sentence_without_comma, LID]) #up_votes, down_votes, age, gender, accent, locale, segment, variant
		sf.write("validation/"+filename, audio_content, 16000)
		n += 1
		if n == n_max_valid:
			break


# export splits
with open("train/metadata.csv","w",newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_train:
		writer.writerow(row)

with open("test/metadata.csv","w",newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_test:
		writer.writerow(row)

with open("validation/metadata.csv","w",newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_validation:
		writer.writerow(row)
'''
# export test files for each language
with open("test_fy-NL.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_test_fy_NL:
		writer.writerow(row)

with open("test_nl.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_test_nl:
		writer.writerow(row)

with open("test_en.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_test_en:
		writer.writerow(row)

with open("test_de.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_test_de:
		writer.writerow(row)

with open("test_sv-SE.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','lid'])
	for row in whole_rows_test_sv_SE:
		writer.writerow(row)


'''






