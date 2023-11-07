import json
import os
import pickle
import sys


def loadfile(filename):
    """ load a file, return a generator. """
    fp = open(filename, 'r', encoding='utf-8')
    for i, line in enumerate(fp):
        yield line.strip('\r\n')
        if i % 100000 == 0:
            print('loading %s(%s)' % (filename, i), file=sys.stderr)
    fp.close()
    print('load %s succ' % filename, file=sys.stderr)


def writetofileLine(data, dfile):
    with open(dfile, 'w') as f:
        for i in range(len(data)):
            item = data[i]
            for j in range(len(item)):
                f.write(str(item[j]) + '\t')
            f.write('\n')


def generate_dataset(filename):
    """ load rating data and split it to training set and test set """
    usernum = 0
    itemnum = 0
    Hisitem = []
    Expitem = []
    User = dict()
    sentence2user = dict()
    usermap = dict()
    itemmap = dict()


    for line in loadfile(filename):
        user, itemLen, ilist, queryNum, queryList = line.split('\t')
        itemlist = ilist.split(',')

        if user in usermap:
            userid = usermap[user]
        else:
            userid = usernum
            usermap[user] = usernum
            User[userid] = []
            usernum += 1
        for i in range(len(itemlist)):
            _item, _showtime, _readtime = itemlist[i].split('|')
            _showtime = int(_showtime)
            _readtime = int(_readtime)
            if (_item in itemmap):
                itemid = itemmap[_item]
            else:
                itemid = itemnum
                itemmap[_item] = itemid
                itemnum += 1

            User[userid].append([itemid, _showtime, _readtime])
    for userid in User.keys():  # 1. order by showtime (up)
        User[userid].sort(key=lambda x: x[1])
    expo_seq = {}
    inter_seq = {}
    for userid in User.keys():  # for each user
        user = User[userid]
        user_expo = [interac[0] for interac in user]
        expo_seq[userid] = user_expo
        user_click = []
        for ans in range(len(user)):  # 2.seek out the answers which have readtime
            _readtime = user[ans][2]
            if _readtime != 0:
                user_click.append(user[ans][0])
        inter_seq[userid] = user_click

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
    for user in expo_seq:
        new_list = []
        for item in expo_seq[user]:
            if item not in item_remap:  # fitter out the items only appeared in exposure data
                # expo_seq[user].remove(item)
                continue
            item = item_remap[item]
            new_list.append(item)
        expo_seq[user] = new_list

    assert len(expo_seq) == len(inter_seq)
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
    print("--- exposure sequence ---")
    maxlen = 0
    minlen = 1000000
    avglen = 0
    for _, ilist in expo_seq.items():
        listlen = len(ilist)
        maxlen = max(maxlen, listlen)
        minlen = min(minlen, listlen)
        avglen += listlen
    avglen /= len(expo_seq)
    print('max length: ', maxlen)
    print('min length: ', minlen)
    print('avg length: ', avglen)
    print('density: ', num_instances / (len(expo_seq) * len(item_remap)))
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
    def writetofile(data, dfile):
        with open(dfile, 'w') as f:
            for items in data:
                for item in items:
                    f.write(str(item) + " ")
                f.write('\n')

    path = './'
    writetofile(train, path + "ori_Train.txt")
    train = augment(train)
    writetofile(train, path + "Train.txt")
    writetofile(val, path + "Valid.txt")
    writetofile(test, path + "Test.txt")
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

    writetofile(expo_train, path + "Exposure_Train.txt")
    writetofile(expo_valid, path + "Exposure_Valid.txt")
    writetofile(expo_test, path + "Exposure_Test.txt")

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
    writetofile(eval_train, path + "Evaluation_Train.txt")
    writetofile(eval_valid, path + "Evaluation_Valid.txt")
    writetofile(eval_test, path + "Evaluation_Test.txt")


if __name__ == '__main__':
    datafile = os.path.join('zhihu1M.txt')
    generate_dataset(datafile)
