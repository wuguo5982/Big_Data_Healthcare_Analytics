import os
import pickle
import pandas as pd

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	converted = ''
	if icd9_str[0].isalpha() and icd9_str[0] == 'E':
		converted = icd9_str[0:4]
	else:
		converted = icd9_str[0:3]
	return converted


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	df_digits = df_icd9['ICD9_CODE'].dropna().apply(transform)
	codemap = {a:b for b, a in enumerate(df_digits.unique())}
	# codemap = {123: 0, 456: 1}
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# Load data from the three csv files
	# Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_diagnosis = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
	df_admission = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))

	# 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	df_diagnosis['ICD9_CODE'] = df_diagnosis['ICD9_CODE'].transform(convert_icd9)
	new_diagnosis = df_diagnosis.copy()
	new_diagnosis = new_diagnosis[df_diagnosis['ICD9_CODE'].apply(lambda x: x in codemap.keys())]
	new_diagnosis['ICD9_CODE'] = new_diagnosis['ICD9_CODE'].apply(lambda x: codemap[x])

	# 3. Group the diagnosis codes for the same visit.
	visit_group = dict(new_diagnosis.groupby("HADM_ID")['ICD9_CODE'].apply(list))


	# 4. Group the visits for the same patient.
	new_admission = df_admission[df_admission["HADM_ID"].apply(lambda x: x in visit_group.keys())]


	# Make a visit sequence dataset as a List of patient Lists of visit Lists
	# Visits for each patient must be sorted in chronological order.
	admission_group = dict(new_admission.sort_values('ADMITTIME').groupby("SUBJECT_ID")['HADM_ID'].apply(list))


	# 6. Make patient-id List and label List also.
	# The order of patients in the three List output must be consistent.

	patient_ids = list(admission_group.keys())
	labels = df_mortality.set_index('SUBJECT_ID').loc[patient_ids]['MORTALITY'].tolist()

	tot_seqs = []
	for y in list(admission_group.values()):
		sub_seq1 = []
		for x in y:
			result = visit_group[x]
			sub_seq1.append(result)
		tot_seqs.append(sub_seq1)
	seq_data = tot_seqs
	# print(len(seq_data))
	# print(len(labels))
	# print(len(seq_data))

	# patient_ids = [0, 1, 2]
	# labels = [1, 0, 1]
	# seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
	return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
