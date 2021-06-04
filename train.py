import torch
import numpy as np
from os.path import dirname, join as pjoin
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from model import CNN_Model, PRNN, PRNNNet

def train(model, device, train_loader, optimizer, num_segment=17):
    model.train()
    model.train_flag = True
    model.num_segment = num_segment
    losses = []
    
    for (data, target) in tqdm(train_loader, leave=True, position=0):
        target = target.reshape(-1,)
        data, target = data.to(device).float(), target.to(device)
        target = target.reshape(1,-1).repeat(model.num_segment,1).T.reshape(-1,)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

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
    parser.add_argument('--lr', type=float, default=1.0e-3,
                        help='learn rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of maximal epochs for training')
    parser.add_argument('--n_epochs_stop', type=int, default=10,
                        help='number of consecutive epochs to decrease learning rate if no improvement of acc of dev dataset')
    parser.add_argument('--train_attempt', type=int, default=1,
                        help='how many times do you want to repeat the training process (unless you would like to study the stability of training usually this number is set to be 1)')
    parser.add_argument('--output_dir', type=str, default='/mnt/hddraid1/NRL_data/output_dir_test/',
                        help='directory to your output')
    parser.add_argument('--model_filename', type=str, default='output_model',
                        help='output model file name')
    parser.add_argument('--outdata_filename', type=str, default='output_data',
                        help='output data file name')
    parser.add_argument('--save', type=bool, default=False,
                        help='option to save the model and output data')
    args = parser.parse_args()

    trainloader_filter = torch.load(args.data_path+'train_dataloader.pth')
    devloader_filter = torch.load(args.data_path+'dev_dataloader.pth')
    
    if args.device_gpu:
        DEVICE = torch.device("cuda:{}".format(args.gpu_idx))
    else:
        DEVICE = torch.device("cpu")
    
    for i in range(args.train_attempt):
        random.seed(args.seed+i)
        np.random.seed(args.seed+i)
        torch.manual_seed(args.seed+i)
        torch.cuda.manual_seed(args.seed+i)
        if args.model == "NRL_CNN":
            net = CNN_Model(device_num=args.device_num).to(DEVICE) # usually use device_num = 30
        elif args.model == "PRNN_CNN":
            net = PRNNNet(device_num=args.device_num).to(DEVICE)
        else:
            raise Exception("Sorry, no matched model. The model should be: NRL_CNN or PRNN_CNN")

        learning_rate = args.lr # it used to be 5e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
#         decayRate = 0.5
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=decayRate)
        training_losses = []
        mean_losses = []
        test_losses = []
        accuracies = []
        EPOCHS = args.epochs
        pred = []
        labels = []
        best_val_acc = 0.0
        epochs_no_improve = 0
        n_epochs_stop = args.n_epochs_stop
        min_val_loss = 1e-2
        early_stop = False
        patience = 0
        
        for epoch in range(EPOCHS):
            training_loss, mean_loss = train(net, DEVICE, trainloader_filter, optimizer)
            test_loss, accuracy, prediction, label = test(net, DEVICE, devloader_filter)
            labels.append(label)
            pred.append(prediction)
            training_losses += training_loss
            mean_losses.append(mean_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"train_attempt: {i}, epoch: {epoch}, mean_loss: {mean_loss}, dev_loss: {test_loss}, accuracy: {accuracy}", flush=True)
            if accuracy <= best_val_acc:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    learning_rate *= 0.5
                    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                    patience += 1
                    print('patience:', patience, 'learning rate:', learning_rate)

                    continue 
            else:
                best_val_acc = accuracy
                epochs_no_improve = 0

        out_data = {"mean_loss": mean_losses, "dev_loss": test_losses, "accuracy": accuracies}
        if args.save:
            PATH_model = args.output_dir 
            PATH_data = args.output_dir 
            model_name = args.model_filename + '_{}_try{}.pth'.format(args.model,i) # 7 devices use table II of NRL's paper
            data_filename = args.outdata_filename + '_{}_try{}.npy'.format(args.model,i)
            torch.save(net.state_dict(), pjoin(PATH_model, model_name))
            np.save(pjoin(PATH_data, data_filename), out_data)