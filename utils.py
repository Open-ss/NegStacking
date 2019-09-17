import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import time
def merge_data(dir_path):
    drug_data = np.load(dir_path+'/drug_data.npy')
    # drug_name = np.load(dir_dir_path+'/drug_name.npy')
    new_label = np.load(dir_path+'/new_label.npy')
    # rngene_data = np.load(dir_path+'/rngene_data.npy')
    rngene_data = np.load(dir_path+'/rngene_data.npy')
    # rngene_name = np.load(dir_path+'/rngene_name.npy')
    # rngene_name_sampled = np.load(dir_path+'/rngene_name_sampled.npy')
    tp_data = np.load(dir_path+'/tp_data.npy')
    # tp_name = np.load(dir_path+'/tp_name.npy')

    data_positive_unlabel = []
    for i in range(drug_data.shape[0]):
        for j in range(tp_data.shape[0]):
            data_positive_unlabel.append(np.concatenate([drug_data[i], tp_data[j]]))

    data_positive_unlabel = np.array(data_positive_unlabel)
    new_label = new_label.T.flatten()
    data_positive = data_positive_unlabel[new_label == 1]
    np.save(dir_path+'/data_merge'+'/data_positive.npy', data_positive)
    data_unlabel = data_positive_unlabel[new_label == 0]
    np.save(dir_path+'/data_merge'+'/data_unlabel.npy', data_unlabel)

    data_negative = []
    for i in range(drug_data.shape[0]):
        for j in range(rngene_data.shape[0]):
            data_negative.append(np.concatenate([drug_data[i], rngene_data[j]]))
    data_negative = np.array(data_negative)
    np.save(dir_path+'/data_merge'+'/data_negative.npy', data_negative)
    pass

# def equalization_sampling_data(dir_path, sample_num):
#     drug_data = np.load(dir_path + '/drug_data.npy')
#     rngene_data_sampled = np.load(dir_path + '/rngene_data_sampled.npy')
#     total_sample_num = drug_data.shape[0] * rngene_data_sampled.shape[0]
#     sampleing_window_size = int(np.sqrt(total_sample_num * 1.0 / sample_num))+1
#     sample_data = []
#
#     remain = sample_num-(drug_data.shape[0] / sampleing_window_size)*(rngene_data_sampled.shape[0] / sampleing_window_size)
#
#     for i in range(drug_data.shape[0]/sampleing_window_size):
#         for j in range(rngene_data_sampled.shape[0] / sampleing_window_size):
#             if i == drug_data.shape[0]/sampleing_window_size - 1:
#                 drug_indexs = np.array(range(i*sampleing_window_size,drug_data.shape[0]))
#             else:
#                 drug_indexs = np.array(range(i*sampleing_window_size, (i+1)*sampleing_window_size))
#
#             if j == rngene_data_sampled.shape[0] / sampleing_window_size - 1:
#                 tp_indexs = np.array(range(j * sampleing_window_size, rngene_data_sampled.shape[0]))
#             else:
#                 tp_indexs = np.array(range(j * sampleing_window_size, (j + 1) * sampleing_window_size))
#             if remain >0:
#                 remain = remain-1
#                 drug_index = np.random.choice(drug_indexs,2,replace=False)
#                 tp_index = np.random.choice(tp_indexs,2,replace=False)
#                 sample_data.append(np.concatenate([drug_data[drug_index[0]], rngene_data_sampled[tp_index[0]]]))
#                 sample_data.append(np.concatenate([drug_data[drug_index[1]], rngene_data_sampled[tp_index[1]]]))
#             else:
#                 drug_index = np.random.choice(drug_indexs)
#                 tp_index = np.random.choice(tp_indexs)
#                 sample_data.append(np.concatenate([drug_data[drug_index], rngene_data_sampled[tp_index]]))
#
#     sample_data = np.array(sample_data)
#     return sample_data
#     pass
def equalization_sampling_data(dir_path, sample_num,random_state=None):
    drug_data = np.load(dir_path + '/drug_data.npy')
    rngene_data = np.load(dir_path + '/rngene_data.npy')
    negative_data_index = [[i, j] for i in range(len(drug_data)) for j in range(len(rngene_data))]
    block_size = int(len(negative_data_index) * 1.0 / sample_num)
    final_negative_data_index = []
    if block_size <= 1:
        if random_state != None:
            np.random.seed(random_state)
        final_negative_data_index = [negative_data_index[i] for i in np.random.choice(len(negative_data_index),sample_num,replace=False)]
    else:
        for i in range(sample_num):
            if random_state != None:
                np.random.seed(random_state+i*10)
            final_negative_data_index.append(negative_data_index[i * block_size + np.random.randint(0, block_size)])
    sample_data = []
    for i,j in final_negative_data_index:
        sample_data.append(np.concatenate([drug_data[i], rngene_data[j]]))
    sample_data = np.array(sample_data)
    return sample_data
def sampling_data_n_sets(data_unlabel, sets_num,sample_num_per_set,random_state=None):
    data_negatives = []
    np.random.seed(random_state)
    data_unlabel_sample = data_unlabel[np.random.choice(range(data_unlabel.shape[0]),sets_num*sample_num_per_set,False)]
    for i in range(sets_num):
        data_negatives.append(data_unlabel_sample[i*sample_num_per_set:(i+1)*sample_num_per_set])
    return data_negatives
# def equalization_sampling_data_n_sets(dir_path,sets_num,sample_num_per_set,random_state=None):
#     sampled_negative = equalization_sampling_data(dir_path=dir_path,sample_num=sample_num_per_set*sets_num,random_state=random_state)
#     data_negatives = []
#     choice_idnexs = []
#     for i in range(sample_num_per_set):
#         np.random.seed(random_state)
#         choice_idnexs.append(np.random.choice(sets_num, sets_num, replace=False) + sets_num * i)
#     choice_idnexs = np.array(choice_idnexs)
#     for i in range(sets_num):
#         sampled_negative_per_learner = sampled_negative[choice_idnexs[:, i].flatten()]
#         np.random.seed(random_state)
#         np.random.shuffle(sampled_negative_per_learner)
#         data_negatives.append(sampled_negative_per_learner)
#     return data_negatives
def equalization_sampling_data_n_sets(dir_path,sets_num,sample_num_per_set,random_state=None):
    sampled_negative = equalization_sampling_data(dir_path=dir_path, sample_num=sample_num_per_set * sets_num,
                                                  random_state=random_state)
    if random_state!=None:
        random_state = random_state+2
    np.random.seed(random_state)
    np.random.shuffle(sampled_negative)
    data_negatives = []
    for i in range(sets_num):
        data_negatives.append(sampled_negative[i*sample_num_per_set:(i+1)*sample_num_per_set])
    return data_negatives
def equalization_sampling_data_n_sets_and_testset_random(dir_path,sets_num,sample_num_per_set,random_state=None):
    sampled_negative,negative_testset = equalization_sampling_data_testset(dir_path=dir_path,sample_num=sample_num_per_set*sets_num,testset_size=sample_num_per_set,random_state=random_state)
    if random_state!=None:
        random_state = random_state+2
    np.random.seed(random_state)
    np.random.shuffle(sampled_negative)
    data_negatives = []
    for i in range(sets_num):
        data_negatives.append(sampled_negative[i*sample_num_per_set:(i+1)*sample_num_per_set])
    return data_negatives,negative_testset
def equalization_sampling_data_n_sets_and_testset_cluster_file(dir_path,sets_num,sample_num_per_set,cluster_ratio=1,random_state=None):
    if cluster_ratio==1:
        return equalization_sampling_data_n_sets_and_testset_random(dir_path=dir_path,sets_num=sets_num,sample_num_per_set=sample_num_per_set,random_state=random_state)
    data_negative, negative_testset = equalization_sampling_data_testset(dir_path=dir_path,
                                                                            sample_num=sample_num_per_set * sets_num*cluster_ratio,
                                                                            testset_size=sample_num_per_set,
                                                                            random_state=random_state)
    data_negative = np.load(dir_path+'/data_merge/'+'data_negative_sampled_cluster.npy')
    data_negatives = []
    for i in range(sets_num):
        data_negatives.append(data_negative[i * sample_num_per_set:(i + 1) * sample_num_per_set])
    return data_negatives, negative_testset

def equalization_sampling_data_n_sets_and_testset_cluster(dir_path,data_positive,sets_num,sample_num_per_set,cluster_ratio=1,random_state=None):
    if cluster_ratio==1:
        return data_positive,equalization_sampling_data_n_sets_and_testset_random(dir_path=dir_path,sets_num=sets_num,sample_num_per_set=sample_num_per_set,random_state=random_state)
    data_negative, negative_testset = equalization_sampling_data_testset(dir_path=dir_path,
                                                                            sample_num=int(sample_num_per_set * sets_num*cluster_ratio),
                                                                            testset_size=sample_num_per_set,
                                                                            random_state=random_state)
    scaler = StandardScaler().fit(np.concatenate([data_positive,data_negative,negative_testset]))
    data_positive = scaler.transform(data_positive)
    negative_testset = scaler.transform(negative_testset)
    data_negative = scaler.transform(data_negative)
    if random_state != None:
        random_state = random_state + 2
        np.random.seed(random_state)
    np.random.shuffle(data_negative)
    data_negatives = []
    # save_sampled_negative = np.zeros((1,data_negative.shape[1]))
    # data_negatives_db = []
    for i in range(sets_num):
        # data_negatives_db.append(data_negative[i*sample_num_per_set:(i+1)*sample_num_per_set])

        data_negative_temp = data_negative[int(i*sample_num_per_set*cluster_ratio):int((i+1)*sample_num_per_set*cluster_ratio)]
        cc = ClusterCentroids(ratio={0: sample_num_per_set})
        sampled_negative,y = cc.fit_sample(np.concatenate([data_negative_temp,np.ones((1,data_negative_temp.shape[1]))]),np.concatenate([np.zeros(data_negative_temp.shape[0]),np.ones(1)]))
        sampled_negative = sampled_negative[y==0]
        data_negatives.append(sampled_negative)
        # save_sampled_negative = np.concatenate([save_sampled_negative,sampled_negative])
    # np.save(dir_path + '/data_merge/data_negative_sampled_cluster.npy',save_sampled_negative[1:])
    return data_positive,(data_negatives,negative_testset)
def equalization_sampling_data_testset(dir_path,sample_num,testset_size,random_state=None):
    drug_data = np.load(dir_path + '/drug_data.npy')
    rngene_data = np.load(dir_path + '/rngene_data.npy')
    negative_data_index = np.array([[i, j] for i in range(len(drug_data)) for j in range(len(rngene_data))])

    np.random.seed(random_state)
    negative_testset_index = np.random.choice(len(negative_data_index),testset_size, False)
    negative_testset = []
    for index in negative_testset_index:
        (i,j) = negative_data_index[index]
        negative_testset.append(np.concatenate([drug_data[i], rngene_data[j]]))
    negative_testset = np.array(negative_testset)

    negative_data_index = negative_data_index[list(set(range(negative_data_index.shape[0])) -
                                                 set(negative_testset_index))]
    block_size = int(len(negative_data_index) * 1.0 / sample_num)
    final_negative_data_index = []
    if block_size <= 1:
        if random_state != None:
            np.random.seed(random_state)
        final_negative_data_index = [negative_data_index[i] for i in np.random.choice(len(negative_data_index),sample_num,replace=False)]
    else:
        for i in range(sample_num):
            if random_state != None:
                np.random.seed(random_state+i)
            final_negative_data_index.append(negative_data_index[i * block_size + np.random.randint(0, block_size)])
    sample_data = []
    for i,j in final_negative_data_index:
        sample_data.append(np.concatenate([drug_data[i], rngene_data[j]]))
    sample_data = np.array(sample_data)
    return sample_data,negative_testset
def get_drug_tp(dir_path):
    drug_data = np.load(dir_path + '/drug_data.npy')
    tp_data_1 = np.load(dir_path + '/tp_data.npy')
    tp_data_2 = np.load(dir_path + '/rngene_data.npy')
    tp_data = np.concatenate([tp_data_1,tp_data_2])
    return drug_data,tp_data

def equalization_sampling_data_n_sets_random(dir_path,sets_num,sample_num_per_set,random_state=None):
    sampled_negative = equalization_sampling_data(dir_path=dir_path,sample_num=sample_num_per_set*sets_num,random_state=random_state)
    if random_state!=None:
        random_state = random_state+2
    np.random.seed(random_state)
    np.random.shuffle(sampled_negative)
    data_negatives = []
    for i in range(sets_num):
        data_negatives.append(sampled_negative[i*sample_num_per_set:(i+1)*sample_num_per_set])
    return data_negatives

def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation]
    # if m % mini_batch_size==0:
    #     num_complete_minibatches = m / mini_batch_size
    # else:
    num_complete_minibatches = int(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_size * num_complete_minibatches:,:]
        mini_batch_Y = shuffled_Y[mini_batch_size * num_complete_minibatches:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
def load_data(negative_sample_num,dataset_name):
    data_positive = np.load('/home/yj/data_bio/Data/after_merge/'+dataset_name+'/data_merge/data_positive.npy')
    data_negative = equalization_sampling_data('/home/yj/data_bio/Data/after_merge/'+dataset_name, negative_sample_num)
    data_unlabel = np.load('/home/yj/data_bio/Data/after_merge/'+dataset_name+'/data_merge/data_unlabel.npy')

    X = np.concatenate([data_positive, data_negative], axis=0)
    y = np.zeros(X.shape[0])
    y[:data_positive.shape[0]] = 1

    index = np.array(range(X.shape[0]))
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    # Standard
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_unlabel = scaler.transform(data_unlabel)
    return X, y, data_unlabel

def load_data2(negative_sample_num,dataset_name):
    data_positive = np.load('/home/yj/data_bio/Data/after_merge/'+dataset_name+'/data_merge/data_positive.npy')
    data_negative = equalization_sampling_data('/home/yj/data_bio/Data/after_merge/'+dataset_name, negative_sample_num)
    X_unlabeled = np.load('/home/yj/data_bio/Data/after_merge/'+dataset_name+'/data_merge/data_unlabel.npy')
    X_labeled = np.concatenate([data_positive, data_negative], axis=0)
    y_labeled = np.zeros(X_labeled.shape[0])
    y_labeled[:data_positive.shape[0]] = 1
    index = np.array(range(X_labeled.shape[0]))
    np.random.shuffle(index)
    X_labeled = X_labeled[index]
    y_labeled = y_labeled[index]
    # Standard
    scaler = StandardScaler()
    scaler.fit(X_labeled)
    X_labeled = scaler.transform(X_labeled)
    X_unlabeled = scaler.transform(X_unlabeled)
    return X_labeled, y_labeled, X_unlabeled