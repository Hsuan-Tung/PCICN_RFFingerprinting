import torch
import numpy as np
from os.path import dirname, join as pjoin
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from model import CNN_Model, PRNN, PRNNNet

def test(model, device, test_loader, num_segment=17, noise_flag=False, snr=40.0):
    model.eval()
    model.train_flag = False
    model.num_segment = num_segment
    test_loss = 0
    correct = 0
    prediction = []
    label = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, leave=True, position=0):
            target = target.reshape(-1,)
            data, target = data.to(device).float(), target.to(device)
            if noise_flag:
                sigma_n = torch.unsqueeze(torch.std(data, dim=1)/(10**(snr/10))/np.sqrt(2), dim=1).repeat(1, data.size(1), 1).to(device)
                noise = torch.unsqueeze(torch.unsqueeze(torch.randn(data.size(1)), dim=0), dim=2).repeat(data.size(0),1,data.size(2)).to(device)
                data = data + sigma_n * noise
            data = data.permute(0,2,1).reshape(data.size(0),2,17,-1)
            output_array = []
            for i in range(num_segment):
                output_array.append(model(data[:,:,i,:]))
            output_array = torch.stack(output_array)
            output = torch.mean(output_array, dim=0)
#             output = model(data)
            test_loss += nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            prediction.append(pred)
            label.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= (len(test_loader.dataset))

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy, prediction, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/hddraid1/NRL_data/dataloader/',
                        help='directory to load models from')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')
    parser.add_argument('--device_gpu', type=bool, default=True,
                        help='using gpu or not')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='gpu index')
    parser.add_argument('--model', type=str, default="NRL_CNN",
                        help='using which model: NRL_CNN or PRNN_CNN')
    parser.add_argument('--device_num', type=int, default=30,
                        help='number of device')
    parser.add_argument('--num_segment', type=int, default=17,
                        help='number of data units in a packet used for training and testing; 17 is the maximum')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--noise_flag', type=bool, default=False,
                        help='add noise to training or not; if True: we will add Gaussian white noise with SNR=10,20,30,40')
    parser.add_argument('--model_path', type=str, default='/mnt/hddraid1/NRL_data/output_dir/',
                        help='directory to your output')
    parser.add_argument('--model_filename', type=str, default='output_model',
                        help='output model file name')
#     parser.add_argument('--outdata_filename', type=str, default='output_data',
#                         help='output data file name')
#     parser.add_argument('--save', type=bool, default=False,
#                         help='option to save the model and output data')
    args = parser.parse_args()

    testloader_filter = torch.load(args.data_path+'test_dataloader.pth')
    
    
    if args.device_gpu:
        DEVICE = torch.device("cuda:{}".format(args.gpu_idx))
    else:
        DEVICE = torch.device("cpu")
        
    if args.model == "NRL_CNN":
        net_test = CNN_Model(device_num=args.device_num).to(DEVICE) # usually use device_num = 30
    elif args.model == "PRNN_CNN":
        net_test = PRNNNet(device_num=args.device_num, input_size=64, hidden_size=16, dt=5).to(DEVICE)
    else:
        raise Exception("Sorry, no matched model. The model should be: NRL_CNN or PRNN_CNN")

    PATH = args.model_path
    model_name = args.model_filename
    net_test.load_state_dict(torch.load(pjoin(PATH, model_name)))
    test_loader = testloader_filter
    test_loss, accuracy, prediction, label = test(net_test, DEVICE, test_loader, noise_flag=args.noise_flag)
    print(f"test_loss: {test_loss}, accuracy: {accuracy}", flush=True)
    
