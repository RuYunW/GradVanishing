import torch
import torch.nn as nn
from model import Net
from tqdm import tqdm
from dataset import BatchData
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages


num_steps = 1024
num_samples = 256
batch_size = 16
lr = 0.1

input_dim = 2
hidden_dim = 10
output_dim = 1
num_hidden_layer = 3



is_plt_grad = False
is_plt_loss = True
is_save_pdf = True    # # #
is_residual = False
seed = 5
hidden_ac_func = 'ReLU'  # 激活函数
ac_func = 'Sigmoid'      # 激活函数
opt = 'SGD'
loss_func = 'MSE'
# hidden_ac_func = 'ReLU'
grad_pic_name = '神经网络中的梯度消失现象（' + hidden_ac_func + '+' + ac_func + '-1, addnorm=' + str(is_residual) + ', seed=' + str(seed) + '）'
grad_save_name = 'grad_vanishing_' + hidden_ac_func + '_' + ac_func + '-1_B=' + str(batch_size) + '_seed=' + str(seed) + 'addnorm=' + str(is_residual) + '.pdf'

loss_pic_name = '神经网络陷入局部极小现象（隐层激活函数ReLU）'
loss_save_name = 'loss='+ loss_func +'+opt='+ opt + '_hidden='+ str(num_hidden_layer) + 'D_hidden='+str(hidden_dim)+'act='+hidden_ac_func+'.pdf'



if ac_func == 'ReLU':
    activation_func = nn.ReLU()
elif ac_func == 'LeakyReLU':
    activation_func = nn.LeakyReLU()
elif ac_func == 'Sigmoid':
    activation_func = nn.Sigmoid()
else:
    pass

if hidden_ac_func == 'ReLU':
    hidden_activation_func = nn.ReLU()
elif hidden_ac_func == 'LeakyReLU':
    hidden_activation_func = nn.LeakyReLU()
elif hidden_ac_func == 'Sigmoid':
    hidden_activation_func = nn.Sigmoid()
else:
    pass

torch.manual_seed(seed)



# 构建数据
data_list = []
for i in range(num_samples):
    flag = torch.rand(1)[0]
    sample_x = torch.rand(input_dim) * (-1) if flag <= 0.5 else torch.rand(input_dim)
    sample_y = torch.zeros(1) if flag <= 0.5 else torch.ones(1)
    data_list.append({'input': sample_x, 'label': sample_y})



dataset = BatchData(data_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Net(input_dim, hidden_dim, output_dim, num_hidden_layer, activation_func, hidden_activation_func, is_residual)
print(model)

if opt == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif opt == 'Adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

if loss_func == 'MSE':
    criterion = nn.MSELoss()
elif loss_func == 'CE':
    criterion = nn.CrossEntropyLoss(reduce=True, weight=None)

model.train()
loss_list = []
loss_list_x = []
grad_list = []
grad_list_log = []
num_epoch = num_steps // batch_size
num_step_per_epoch = num_samples // batch_size
for step in tqdm(range(num_steps)):
    if step % num_step_per_epoch == 0:
        iter_data = iter(dataloader)
    data = next(iter_data)
    output = model(data)
    if loss_func == 'CE':
        loss = criterion(output, torch.tensor(data['label'].reshape(-1), dtype=torch.long))
    else:
        loss = criterion(output, data['label'])

    if step % (num_steps/1024) == 0:
        loss_list_x.append(step)
        loss_list.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 5 and is_plt_grad:
        for i, (name, parms) in enumerate(model.named_parameters()):
            if i % 2 == 1:
                grad_list.append(torch.mean(parms.grad))
                grad_list_log.append(torch.log(torch.mean(parms.grad)))
        print(data['label'])
        print(output)
        break

if is_plt_grad:
    plt.plot(grad_list)
    plt.plot(grad_list_log)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(['mean grad', 'log grad'])
    plt.xlabel('网络层数')
    plt.ylabel('梯度')
    plt.title(grad_pic_name)
    if is_save_pdf:
        pp = PdfPages(grad_save_name)
        pp.savefig()
        pp.close()
        plt.close()
    else:
        plt.show()

if is_plt_loss:
    plt.plot(loss_list_x, loss_list)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('迭代步数（Step）')
    plt.ylabel('Loss')
    plt.title(loss_pic_name)
    if is_save_pdf:
        pp = PdfPages(loss_save_name)
        pp.savefig()
        pp.close()
        plt.close()
    else:
        plt.show()





