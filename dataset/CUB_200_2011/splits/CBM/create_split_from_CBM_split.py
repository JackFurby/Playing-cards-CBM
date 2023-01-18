import pickle
import os
import glob

val_split = pickle.load(open("./val.pkl", "rb"))
train_split = pickle.load(open("./train.pkl", "rb"))
test_split = pickle.load(open("./test.pkl", "rb"))

splits = [[train_split, "train"], [val_split, "val"], [test_split, "test"]]


for i in splits:
	updated_split = {}
	for sample in i[0]:
		updated_split[sample['id']] = {
			"img_path": f"imgs/CBM/{sample['img_path']}",
			"class_label": sample['class_label'] - 1,
			"concept_label": sample["attribute_label"],
			"card_points": []
		}


	pickle.dump(updated_split, open(f"./{i[1]}.pkl", "wb"))



"""
{
	id:
	img_path:
	class_label:
	attribute_label:,
	attribute_certainty:,
	locs:
}
"""



"""
pkl / json file structure

id:
{
	img_path: string
	class_label: int (0 indexed)
	concept_label: [list of 0s and 1s]
	card_points: [[List of card corner coordinates]]
}

"""
