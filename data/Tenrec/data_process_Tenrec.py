import csv

# Please download 'QB-video.csv' of the Tenrec dataset.
filename = 'QB-video.csv'  # user_id,item_id,click,follow,like,share,video_category,watching_times,gender,age
user_map = dict()
item_map = dict()
inter_seq = dict()  # interaction sequence
expo_seq = dict()  # exposure data sequence
user_count = dict()
item_count = dict()  # only cal the count of click event
user_tot = item_tot = 1  # id of user and item both start at 1
with open(filename) as csvfile:  # get user_count and item_count
    csv_reader = csv.reader(csvfile)  # read file using csv
    csv_header = next(csv_reader)
    for row in csv_reader:
        user_id = int(row[0])
        item_id = int(row[1])
        click = int(row[2])
        if user_id not in user_map:
            user_map[user_id] = user_tot
            user_count[user_id] = 0
            user_tot += 1
            # inter_seq[user_id] = []
            # expo_seq[user_id] = []
        if item_id not in item_map:
            item_map[item_id] = item_tot
            item_count[item_id] = 0
            item_tot += 1
        if click == 1:
            item_count[item_id] += 1
            user_count[user_id] += 1

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # read file using csv
    csv_header = next(csv_reader)
    for row in csv_reader:
        user_id = int(row[0])
        item_id = int(row[1])
        click = int(row[2])
        # filter out these users and items in inter_seq and exposure data
        if user_count[user_id] < 2 or item_count[item_id] < 5:
            continue
        user_id = user_map[user_id]
        item_id = item_map[item_id]
        if user_id not in inter_seq:
            inter_seq[user_id] = []
            expo_seq[user_id] = []
        if click == 1:
            inter_seq[user_id].append(item_id)
        expo_seq[user_id].append(item_id)  # click + unclick

tmp_inter_seq = dict()
for user in inter_seq:
    nfeedback = len(inter_seq[user])
    if nfeedback > 50:
        inter_seq[user] = inter_seq[user][:50]
    if nfeedback >= 2:
        tmp_inter_seq[user] = inter_seq[user]
    else:
        expo_seq.pop(user)
inter_seq = tmp_inter_seq.copy()
for user in expo_seq:
    nfeedback = len(expo_seq[user])
    if nfeedback > 200:
        expo_seq[user] = expo_seq[user][:200]

# remapping item id
item_remap = dict()
new_item_id = 1
for user in inter_seq:
    new_list = []
    for item in inter_seq[user]:
        if item not in item_remap:
            item_remap[item] = new_item_id
            new_item_id += 1
        item = item_remap[item]
        new_list.append(item)
    inter_seq[user] = new_list
for user in expo_seq:  # fitter out the items only appeared in exposure data
    new_list = []
    for item in expo_seq[user]:
        if item not in item_remap:
            # expo_seq[user].remove(item)
            continue
        item = item_remap[item]
        new_list.append(item)
    expo_seq[user] = new_list

num_instances = sum([len(ilist) for _, ilist in inter_seq.items()])
print('total user: ', len(inter_seq))
print('total instances: ', num_instances)
print('total items: ', len(item_remap))

print("--- click sequence ---")
maxlen = 0
minlen = 1000000
avglen = 0
for _, ilist in inter_seq.items():
    listlen = len(ilist)
    maxlen = max(maxlen, listlen)
    minlen = min(minlen, listlen)
    avglen += listlen
avglen /= len(inter_seq)
print('max length: ', maxlen)
print('min length: ', minlen)
print('avg length: ', avglen)
print('density: ', num_instances / (len(inter_seq) * len(item_remap)))

print("--- exposure data ---")
maxlen = 0
minlen = 1000000
avglen = 0
for _, ilist in expo_seq.items():
    listlen = len(ilist)
    maxlen = max(maxlen, listlen)
    minlen = min(minlen, listlen)
    avglen += listlen
avglen /= len(inter_seq)
print('max length: ', maxlen)
print('min length: ', minlen)
print('avg length: ', avglen)
print('density: ', num_instances / (len(inter_seq) * len(item_remap)))

# split dataset and write file
inter_seq = [inter_seq[items] for items in inter_seq]  # dict2list
expo_seq = [expo_seq[items] for items in expo_seq]
data_len = len(inter_seq)
train_idx = int(data_len * 0.8)
val_idx = train_idx + int(data_len * 0.1)
test_idx = val_idx + int(data_len * 0.1)

train_indices = [i for i in range(1, int(train_idx))]
valid_indices = [i for i in range(train_idx, val_idx)]
test_indices = [i for i in range(val_idx, test_idx)]

train = [inter_seq[i] for i in train_indices]
val = [inter_seq[i] for i in valid_indices]
test = [inter_seq[i] for i in test_indices]


# train data augmentation
def augment(seq_list):
    u2seq = []
    for seq in seq_list:
        for i in range(2, len(seq) + 1):
            u2seq.append(seq[:i])
    return u2seq



# train = train

def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for items in data:
            for item in items:
                f.write(str(item) + " ")
            f.write('\n')


path = './tenrec_qbv/'
writetofile(train, path + "ori_Train.txt")
train = augment(train)
writetofile(train, path + "Train.txt")
writetofile(val, path + "valid.txt")
writetofile(test, path + "test.txt")
aug_test = augment(test)
writetofile(aug_test, path + "session_Test.txt")

# 70% exposure data to train the exposure model and remain 30% to evaluation the performance of debias

# 70% data for training exposure model
expo_idx = int(data_len * 0.7)
expo_indices = [i for i in range(1, int(expo_idx))]
expo = [expo_seq[i] for i in expo_indices]

train_idx = int(expo_idx * 0.8)
val_idx = train_idx + int(expo_idx * 0.1)
test_idx = val_idx + int(expo_idx * 0.1)
train_indices = [i for i in range(1, int(train_idx))]
valid_indices = [i for i in range(train_idx, val_idx)]
test_indices = [i for i in range(val_idx, test_idx)]
expo_train = [expo[i] for i in train_indices]
writetofile(expo_train, path + "ori_Exposure_Train.txt")
expo_train = augment(expo_train)
expo_valid = [expo[i] for i in valid_indices]
expo_test = [expo[i] for i in test_indices]

# writetofile(expo_train, path + "Exposure_Train.txt")
# writetofile(expo_valid, path + "Exposure_Valid.txt")
# writetofile(expo_test, path + "Exposure_Test.txt")

eval_idx = data_len
eval_indices = [i for i in range(expo_idx, data_len)]
eva = [expo_seq[i] for i in eval_indices]

eval_len = data_len * 0.3
train_idx = int(eval_len * 0.8)
val_idx = train_idx + int(eval_len * 0.1)
test_idx = val_idx + int(eval_len * 0.1)
train_indices = [i for i in range(1, int(train_idx))]
valid_indices = [i for i in range(train_idx, val_idx)]
test_indices = [i for i in range(val_idx, test_idx)]
eval_train = [eva[i] for i in train_indices]
writetofile(eval_train, path + "ori_Evaluation_Train.txt")
eval_train = augment(eval_train)
eval_valid = [eva[i] for i in valid_indices]
eval_test = [eva[i] for i in test_indices]
# writetofile(eval_train, path + "Evaluation_Train.txt")
# writetofile(eval_valid, path + "Evaluation_Valid.txt")
# writetofile(eval_test, path + "Evaluation_Test.txt") 
