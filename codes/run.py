import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from myautoencoder import myAutoencoder
from mymlp import myMLP
from myattention import myAttention,myMHAttention
import itertools



M = 8
d_modl = 64
N = 1
d_ff = 512
lr = 0.1
lr_autoencoder = 1e-4
EPOCH = 100
use_MHA = False
is_train = True
dataset="Movies"
BATCH_SIZE = 500

# import myautoencoder

def readdata(file_path):
    data = pd.read_csv(file_path, header=None, sep=' ')
    # print(data)
    user = list(data[0])
    item = list(data[1])
    rating = list(data[2])
    user_num = max(user) + 1
    item_num = max(item) + 1
    rating_matrix = sparse.coo_matrix((rating, (user, item)), shape=(user_num, item_num))
    rating_matrix = rating_matrix.todense()
    # print(user_num,item_num)
    return rating_matrix, user_num, item_num, data


def get_I(data):
    I = data > 0
    I=I.int()
    return I


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def res_to_file(r_test, u_test, filepath):
    indices = u_test.nonzero().cpu()
    # print(indices[0],indices[1])
    # print(y[indices])
    # print(y_hat[indices])
    d = np.vstack((indices[0], indices[1], u_test[indices], r_test[indices])).T
    np.savetxt(filepath, d)
    return d


def decoder_to_file(decoder, rating, filepath):
    decoder = decoder.detach().numpy()
    rating = rating.detach().numpy()
    dis = decoder - rating
    np.savetxt(filepath, dis)
    return dis


def get_rmse(res, rating):
    # res = res.detach().numpy()
    # rating = rating.detach().numpy()
    I = rating > 0
    I = I.int()
    
    # res = list(res)
    # print(res)
    res[res < 0] = 0
    res[res > 5] = 5
    # print(res)
    res = np.array(res)
    res = res*I
    num = I.sum()
    dis = res - rating.float()
    dis = torch.mul(dis, dis)
    rmse = (dis.sum()) / num
    rmse = torch.sqrt(rmse)
    return rmse


def get_mae(res, rating):
    # res = res.detach().numpy()
    I = rating > 0
    I=I.int()
    
    # res = list(res)
    res[res < 0] = 0
    res[res > 5] = 5
    res = np.array(res)
    res = res*I
    num = I.sum()
    dis = abs(res - rating.float())
    mae = (dis.sum()) / num
    return mae


def my_sigmoid(x, a):
    x = torch.sigmoid(x)
    return x * a


def pre_run(file_path1, file_path2, LRu, LRi):
    # 超参数
    EPOCH = 50
    
    # LRu = 0.0002
    # LRi = 0.0002
    
    tuple_1 = readdata(file_path1)
    rating_1 = tuple_1[0]
    #
    rating_1T = rating_1.T
    user1_num = tuple_1[1]
    item1_num = tuple_1[2]
    train_data = tuple_1[3]
    tuple_2 = readdata(file_path2)
    rating_2 = tuple_2[0]
    user2_num = tuple_2[1]
    item2_num = tuple_2[2]
    
    if item1_num > item2_num:
        t = item1_num - item2_num
        temp = np.zeros((user2_num, t))
        rating_2 = np.c_[rating_2, temp]
        item2_num = item1_num
    else:
        t = item2_num - item1_num
        temp = np.zeros((user1_num, t))
        rating_1 = np.c_[rating_1, temp]
        item1_num = item2_num
    
    if user1_num > user2_num:
        t = user1_num - user2_num
        temp = np.zeros((t, item2_num))
        rating_2 = np.r_[rating_2, temp]
        user2_num = user1_num
    else:
        t = user2_num - user1_num
        temp = np.zeros((t, item1_num))
        rating_1 = np.r_[rating_1, temp]
        user1_num = user2_num
    
    print('r1:', rating_1.shape, 'r2:', rating_2.shape)
    
    # autoencoder
    ae_user_1 = myAutoencoder(item1_num)
    ae_item_1 = myAutoencoder(user1_num)
    optim_user1 = torch.optim.SGD(ae_user_1.parameters(), lr=LRu, momentum=0.9)
    optim_item1 = torch.optim.SGD(ae_item_1.parameters(), lr=LRi, momentum=0.9)
    # ae_item_1 = myAutoencoder(user1_num)
    # optim_item1 = torch.optim.Adam(ae_item_1.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    

    rating_1T = rating_1.T
    rating_2T = rating_2.T
    user1_dataset = Data.TensorDataset(torch.from_numpy(rating_1).cuda(), torch.from_numpy(rating_1).cuda())
    item1_dataset = Data.TensorDataset(torch.from_numpy(rating_1T).cuda(), torch.from_numpy(rating_1T).cuda())
    # 把 dataset 放入 DataLoader
    loader_user1 = Data.DataLoader(
        dataset=user1_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  
    )
    
    loader_item1 = Data.DataLoader(
        dataset=item1_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  
    )
    
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader_user1):
            x = x.float()
            y = y.float()
            I = get_I(y)
            y = y.mul(I.float())
            encoded, decoded = ae_user_1(x)
            loss = loss_func(decoded, y)  # mean square error
            optim_user1.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optim_user1.step()  # apply gradients
            # print(loss.detach().numpy())
    
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader_item1):
            x = x.float()
            y = y.float()
            I = get_I(y)
            y = y.mul(I.float())
            encoded, decoded = ae_item_1(x)
            loss = loss_func(decoded, y)  # mean square error
            optim_item1.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optim_item1.step()  # apply gradients
            # print(loss.detach().numpy())
    
    u1_encoder, u1_decoder = ae_user_1(torch.from_numpy(rating_1).float().cuda())
    Iu = get_I(torch.from_numpy(rating_1).cuda())
    u1_decoder = u1_decoder.float()
    u1_decoder = u1_decoder.mul(Iu.float())
    rmseu, maeu = get_rmse(u1_decoder.detach().numpy(), rating_1), get_mae(u1_decoder.detach().numpy(), rating_1)
    i1_encoder, i1_decoder = ae_item_1(torch.from_numpy(rating_1T).float().cuda())
    Ii = get_I(torch.from_numpy(rating_1T).cuda())
    i1_decoder = i1_decoder.float()
    i1_decoder = i1_decoder.mul(Ii.float())
    rmsei, maei = get_rmse(i1_decoder.detach().numpy(), rating_1T), get_mae(i1_decoder.detach().numpy(), rating_1T)
    
    return u1_encoder, u1_decoder, i1_encoder, i1_decoder, torch.from_numpy(rating_1).cuda(), torch.from_numpy(rating_2).cuda()




def run_myMLP_att(su1_encoder, su2_encoder, su3_encoder, tu_encoder, ti_encoder, lr, s1_train, s2_train, s3_train,
                  t_train, t_test):
    u1mlp = myMLP().cuda()
    u2mlp = myMLP().cuda()
    u3mlp = myMLP().cuda()
    u4mlp = myMLP().cuda()
    imlp = myMLP().cuda()
    if use_MHA==True:
        weight = myMHAttention(N,M,d_modl,d_ff).cuda()
    else:
        weight = myAttention().cuda()
    optim = torch.optim.SGD([{'params': list(u1mlp.parameters()) + list(u2mlp.parameters()) + list(
        u3mlp.parameters()) + list(u4mlp.parameters()) + list(imlp.parameters()) + list(weight.parameters())}], lr=lr)
    U = su1_encoder.size()[0]
    # print('U:',U)
    index_u = [i for i in range(U)]
    I = ti_encoder.size()[0]
    index_i = [i for i in range(I)]
    tempu = torch.tensor(index_u).view(-1, 1).cuda()
    user_dataset = Data.TensorDataset(su1_encoder, su2_encoder, su3_encoder, tu_encoder, tempu)
    tempi = torch.tensor(index_i).view(-1, 1).cuda()
    item_dataset = Data.TensorDataset(ti_encoder, tempi)
    
    user = Data.DataLoader(
        dataset=user_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  
    )
    item = Data.DataLoader(
        dataset=item_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  
    )
    
    for epoch in range(EPOCH):
        
        u1_vec = u1mlp(su1_encoder.float())
        u2_vec = u2mlp(su2_encoder.float())
        u3_vec = u3mlp(su3_encoder.float())
        u4_vec = u4mlp(tu_encoder.float())
        i_vec = imlp(ti_encoder.float()) 
        u = weight(u1_vec, u2_vec, u3_vec, u4_vec)
        r_test = torch.mm(u, i_vec.permute(1,0)) 
        r_test = my_sigmoid(r_test, 5)
        
        I = get_I(t_train)
        r_test = r_test.mul(I.float())
        rmse, mae = get_rmse(r_test, t_train), get_mae(r_test,t_train)
        print(epoch, 'rmse:', rmse.item(), 'mae:', mae.item())
        for step in range(int(U / BATCH_SIZE)):
            user_loader = iter(cycle(user))
            item_loader = iter(cycle(item))
            u1, u2, u3, u4, ind_u = next(user_loader)
            i, ind_i = next(item_loader)
            u1_vec = u1mlp(u1)
            u2_vec = u2mlp(u2)
            u3_vec = u3mlp(u3)
            u4_vec = u4mlp(u4)
            i_vec = imlp(i)
            u = weight(u1_vec, u2_vec, u3_vec, u4_vec)
            rating = torch.mm(u, i_vec.permute(1,0))
            # if mae > 5:
            rating = my_sigmoid(rating, 5)
            
            temp_train = t_train
            y = np.zeros((BATCH_SIZE, BATCH_SIZE))
            for i in range(BATCH_SIZE):
                for j in range(BATCH_SIZE):
                    y[i][j] = temp_train[ind_u[i][0]][ind_i[j][0]]
            y = torch.from_numpy(y).cuda()
            I = get_I(y)
            rating = rating.mul(I.float())
            # print(rating)
            loss_func = nn.MSELoss()
            loss = loss_func(rating.float(), y.float())
            optim.zero_grad()  # clear gradients for this training step
            loss.backward(retain_graph=True)  # backpropagation, compute gradients
            optim.step()  # apply gradients
        # print(loss.detach().numpy())
    # print(epoch, temp.detach().numpy())
    
    u1_vec = u1mlp(su1_encoder)
    u2_vec = u2mlp(su2_encoder)
    u3_vec = u3mlp(su3_encoder)
    u4_vec = u4mlp(tu_encoder)
    i_vec = imlp(ti_encoder)
    u = weight(u1_vec, u2_vec, u3_vec, u4_vec)
    rtest = torch.mm(u, i_vec.permute(1,0))
    rtest = my_sigmoid(rtest, 5)
    I = get_I(t_test)
    rtest = rtest.mul(I.float())
    return rtest


def save_autoencoder_tensor():
    read_path = "./data/"
    save_path = "./data/amazon_autoencoder_data/"
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path))
    u1_encoder, i1_encoder, u1_train, u1_test = pre_run(read_path + "Books_train.csv", read_path + "Books_test.csv",
                                                        lr_autoencoder, lr_autoencoder, "Books", False)
    save_info = {'u1': u1_encoder, 'i1': i1_encoder, 'train': u1_train, 'test': u1_test}
    torch.save(save_info, save_path + "Books")
    
    u2_encoder, i2_encoder, u2_train, u2_test = pre_run(read_path + "CDs_and_Vinyl_train.csv",
                                                        read_path + "CDs_and_Vinyl_test.csv", lr_autoencoder,
                                                        lr_autoencoder,
                                                        "CDs", False)
    save_info = {'u2': u2_encoder, 'i2': i2_encoder, 'train': u2_train, 'test': u2_test}
    torch.save(save_info, save_path + "CDs")

    u3_encoder, i3_encoder, u3_train, u3_test = pre_run(read_path + "Digital_Music_train.csv",
                                                        read_path + "Digital_Music_test.csv", lr_autoencoder,
                                                        lr_autoencoder,
                                                        "Music", False)
    save_info = {'u3': u3_encoder, 'i3': i3_encoder, 'train': u3_train, 'test': u3_test}
    torch.save(save_info, save_path + "Music")
    
    u4_encoder, i4_encoder, u4_train, u4_test = pre_run(read_path + "Movies_and_TV_train.csv",
                                                        read_path + "Movies_and_TV_test.csv", lr_autoencoder,
                                                        lr_autoencoder,
                                                        "Movies", False)
    save_info = {'u4': u4_encoder, 'i4': i4_encoder, 'train': u4_train, 'test': u4_test}
    torch.save(save_info, save_path + "Movies")

def load_autoencoder_tensor(dataset):
    load_path = "./data/amazon_autoencoder_data/"
    if dataset=="Books":
        medial_tensor = torch.load(load_path + "Books")
        u1_encoder, i1_encoder, u1_train, u1_test = medial_tensor['u1'], medial_tensor['i1'], medial_tensor['train'],medial_tensor['test']
        medial_tensor = torch.load(load_path + "CDs")
        u2_encoder= medial_tensor['u2']
        medial_tensor = torch.load(load_path + "Music")
        u3_encoder = medial_tensor['u3']
        medial_tensor = torch.load(load_path + "Movies")
        u4_encoder= medial_tensor['u4']
        return u2_encoder, u3_encoder, u4_encoder, u1_encoder, i1_encoder,u1_train, u1_test
    elif dataset=="CDs":
        medial_tensor = torch.load(load_path + "Books")
        u1_encoder = medial_tensor['u1']
        medial_tensor = torch.load(load_path + "CDs")
        u2_encoder,i2_encoder, u2_train, u2_test= medial_tensor['u2'],medial_tensor['i2'],medial_tensor['train'],medial_tensor['test']
        medial_tensor = torch.load(load_path + "Music")
        u3_encoder = medial_tensor['u3']
        medial_tensor = torch.load(load_path + "Movies")
        u4_encoder = medial_tensor['u4']
        return u3_encoder, u4_encoder, u1_encoder, u2_encoder, i2_encoder,u2_train, u2_test
    elif dataset=="Music":
        medial_tensor = torch.load(load_path + "Books")
        u1_encoder = medial_tensor['u1']
        medial_tensor = torch.load(load_path + "CDs")
        u2_encoder = medial_tensor['u2']
        medial_tensor = torch.load(load_path + "Music")
        u3_encoder, i3_encoder, u3_train, u3_test = medial_tensor['u3'],medial_tensor['i3'],medial_tensor['train'],medial_tensor['test']
        medial_tensor = torch.load(load_path + "Movies")
        u4_encoder = medial_tensor['u4']
        return u4_encoder, u1_encoder, u2_encoder, u3_encoder, i3_encoder, u3_train, u3_test
    elif dataset=="Movies":
        medial_tensor = torch.load(load_path + "Books")
        u1_encoder = medial_tensor['u1']
        medial_tensor = torch.load(load_path + "CDs")
        u2_encoder = medial_tensor['u2']
        medial_tensor = torch.load(load_path + "Music")
        u3_encoder = medial_tensor['u3']
        medial_tensor = torch.load(load_path + "Movies")
        u4_encoder, i4_encoder, u4_train, u4_test = medial_tensor['u4'],medial_tensor['i4'], medial_tensor['train'],medial_tensor['test']
        return u1_encoder, u2_encoder, u3_encoder, u4_encoder, i4_encoder,u4_train, u4_test

 
if __name__ == '__main__':
    path="./data/"
    pre_run(path + "Books_train.csv", path + "Books_test.csv", lr_autoencoder, lr_autoencoder,"Books",is_train)
    pre_run(path + "CDs_and_Vinyl_train.csv",path + "CDs_and_Vinyl_test.csv", lr_autoencoder,lr_autoencoder, "CDs",is_train)
    pre_run(path + "Digital_Music_train.csv", path + "Digital_Music_test.csv", lr_autoencoder, lr_autoencoder, "Music",is_train)
    pre_run(path + "Movies_and_TV_train.csv", path + "Movies_and_TV_test.csv", lr_autoencoder, lr_autoencoder, "Movies",is_train)
    print('finished')
    
    save_autoencoder_tensor()
    

    if use_MHA:
        print("run "+dataset+" MHA without ff  with MLP+MHAtt N={} M={} d_model={} lr={:.6f} epoch={}".format(N,M,d_modl,lr,EPOCH))
    else:
        print("run "+dataset+" with Att d_model={} lr={:.6f} epoch={}".format(d_modl,lr,EPOCH))

    if dataset=="Books":
        u2_encoder, u3_encoder, u4_encoder, u1_encoder, i1_encoder,u1_train, u1_test=load_autoencoder_tensor(dataset)
        r_test1 = run_myMLP_att(u2_encoder, u3_encoder, u4_encoder, u1_encoder, i1_encoder, lr,  u1_train, u1_test)
        rmse1, mae1 = get_rmse(r_test1, u1_test), get_mae(r_test1, u1_test)
    elif dataset=="CDs":
        u3_encoder, u4_encoder, u1_encoder, u2_encoder, i2_encoder,u2_train, u2_test=load_autoencoder_tensor(dataset)
        r_test2 = run_myMLP_att(u3_encoder, u4_encoder, u1_encoder, u2_encoder, i2_encoder, lr, u2_train, u2_test)
        rmse2, mae2 = get_rmse(r_test2, u2_test), get_mae(r_test2, u2_test)
    elif dataset=="Music":
        u4_encoder, u1_encoder, u2_encoder, u3_encoder, i3_encoder,u3_train, u3_test=load_autoencoder_tensor(dataset)
        r_test3 = run_myMLP_att(u4_encoder, u1_encoder, u2_encoder, u3_encoder, i3_encoder, lr,  u3_train, u3_test)
        rmse3, mae3 = get_rmse(r_test3, u3_test), get_mae(r_test3, u3_test)
    elif dataset=="Movies":
        u1_encoder, u2_encoder, u3_encoder, u4_encoder, i4_encoder,u4_train, u4_test=load_autoencoder_tensor(dataset)
        r_test4 = run_myMLP_att(u1_encoder, u2_encoder, u3_encoder, u4_encoder, i4_encoder, lr,  u4_train, u4_test)
        rmse4, mae4 = get_rmse(r_test4, u4_test), get_mae(r_test4, u4_test)
    print('finished')