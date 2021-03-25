import json
import torch
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import rfftn

from metric_dreem import dreem_sleep_apnea_custom_metric
from models import Force_connex

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


def format_prediction_to_submission_canvas(prediction_dict):
    test_result_list = []
    for sample_id, pred in prediction_dict.items():
        test_result_list.append([sample_id] + pred)
    test_result_df = pd.DataFrame(test_result_list, columns=['ID']+[f'y_{i}' for i in range(90)])
    return test_result_df

# def compute_conv2d_output_dim(input_dim, kernels=[], strides=[], net='CMCMCC'):
#     if net == 'CMCMCC':


def train_model(epoch, model, train_loader, optimizer, criterion, params, threshold=0.5):

    epoch_loss = 0
    epoch_acc = 0
    epoch_acc_pp = 0
    model.train()
    fcnx = Force_connex(params)

    with tqdm(train_loader,
              desc=(f'Train - Epoch: {epoch}'),
              unit=' patient',
              ncols=80,
              unit_scale=params.batch_size) as t:

        for i, (signal, target) in enumerate(t):

            optimizer.zero_grad()
            signal = signal.type(torch.FloatTensor)
            signal, target = signal.cuda(), target.cuda()

            preds = model(signal)
            preds = preds.type(torch.FloatTensor).cpu()
            target = target.type(torch.FloatTensor).cpu()
            loss = criterion(preds, target)
            if params.loss_weighting:
                weight = torch.ones(target.size(), dtype=torch.float64) + params.pen_apnea*target
                loss = loss * weight
                loss = loss.mean()

            acc = dreem_sleep_apnea_custom_metric((preds.detach()>threshold).float(), (target.detach()>pow(10,-5)).float())
            if params.post_process:
                with torch.no_grad():
                    preds_pp = fcnx(preds,threshold)
                    acc_pp = dreem_sleep_apnea_custom_metric((preds_pp.detach()>threshold).float(), (target.detach()>pow(10,-5)).float())
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_acc_pp += acc_pp

    if params.post_process:
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader), epoch_acc_pp / len(train_loader)
    else:
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def evaluate_model(epoch, model, val_loader, criterion, params, threshold=0.5):

    epoch_loss = 0
    epoch_acc = 0
    epoch_acc_pp = 0
    model.eval()
    fcnx = Force_connex(params)

    with tqdm(val_loader,
             desc=(f'Validation - Epoch: {epoch}'),
             unit=' patient',
             ncols=80,
             unit_scale=params.batch_size) as t:

        with torch.no_grad():

            for i, (signal, target) in enumerate(t):

                signal = signal.type(torch.FloatTensor)
                signal, target = signal.cuda(), target.cuda()

                preds = model(signal)
                preds = preds.type(torch.FloatTensor).cpu()
                target = target.type(torch.FloatTensor).cpu()
                loss = criterion(preds, target)
                if params.loss_weighting:
                    weight = torch.ones(target.size(), dtype=torch.float64) + params.pen_apnea*target
                    loss = loss * weight
                    loss = loss.mean()
                
                acc = dreem_sleep_apnea_custom_metric((preds.detach()>threshold).float(), (target.detach()>pow(10,-5)).float())
                if params.post_process:
                    preds_pp = fcnx(preds,threshold)
                    acc_pp = dreem_sleep_apnea_custom_metric((preds_pp.detach()>threshold).float(), (target.detach()>pow(10,-5)).float())

                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_acc_pp += acc_pp

    if params.post_process:
        return epoch_loss / len(val_loader), epoch_acc / len(val_loader), epoch_acc_pp / len(val_loader)
    else:
        return epoch_loss / len(val_loader), epoch_acc / len(val_loader)



def test_model(model, test_loader, params, threshold=0.5):

    model.eval()
    preds_dict = {}
    fcnx = Force_connex(params)

    with tqdm(test_loader,
             desc=(f'Test: '),
             unit=' patient',
             ncols=80,
             unit_scale=params.test_batch_size) as t:

        with torch.no_grad():

            for i, (signal, sample_index, subject_index) in enumerate(t):

                signal = signal.type(torch.FloatTensor)
                signal = signal.cuda()
                preds = model(signal).unsqueeze(0)
                preds = preds.type(torch.FloatTensor).cpu()
                sample_index = sample_index.item()
                if params.post_process:
                    preds_pp = fcnx(preds,threshold)
                    preds_pp = preds_pp.squeeze(0)
                    preds_dict[int(sample_index)] = [int(x>threshold) for x in preds_pp.tolist()]
                else:
                    preds_dict[int(sample_index)] = [int(x>threshold) for x in preds.tolist()]

    preds_df = format_prediction_to_submission_canvas(preds_dict)
    return preds_df


def plot_curves(train_losses, train_accuracies, validation_losses, validation_accuracies, params):

    x = range(params.nepochs)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, train_losses, label='train', color='#6F1BDA')
    axs[0].plot(x, validation_losses, label='val', color='#DA1BC6')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    axs[1].plot(x, train_accuracies, label='train', color='#6F1BDA')
    axs[1].plot(x, validation_accuracies, label='val', color='#DA1BC6')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('acc')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('curves.pdf')
    plt.show()


def replace_tuple_at_index(tup, ix, val):
    lst = list(tup)
    lst[ix] = val
    return tuple(lst)


def normalize_apnea_data(x, channel_axis=0, window_axis=1, sampling_axis=2):
    '''
    inputs:
        - x (size: (n_signals, window_length, sampling_freq)):
        the array of n_signals PSG signals with sampling frequency sampling_freq (Hz)
        over window_length seconds
        - channel_axis (integer): ...
        - window_axis (integer): ...
        - sampling_axis (integer): ...
        output:
        - x_normalized (size: (n_signals, windown_length, sampling_freq): the input array, normalized
    '''
    assert len(x.shape) == 3, "x should be a 3d array"
    adapted_shape = replace_tuple_at_index((1,1,1), channel_axis, -1)
    x_avg = np.mean(x, axis=(window_axis,sampling_axis)).reshape(adapted_shape)
    x_std = np.std(x, axis=(window_axis,sampling_axis)).reshape(adapted_shape)
    x_normalized = (x-x_avg) / x_std
    return x_normalized


def compute_FFT_features(x, max_order=-1, channel_axis=0, window_axis=1, sampling_axis=2):
    '''
    inputs:
        - x (size: (n_signals, window_length, sampling_freq)):
        the array of 8 PSG signals with sampling frequency 100Hz
        over 90 seconds for nb_samples samples
        - max_order (integer): the maximum order of Fourier coefficients considered
        output:
        - x_fft (size: (n_signals, windown_length, max_order): the representation of the sequence in the Fourier domain
        truncated at max_order
    '''
    x_normalized = normalize_apnea_data(x, channel_axis, window_axis, sampling_axis)
    x_fft = rfftn(x_normalized, axes=[sampling_axis])
    if max_order == -1:
        idxs = range(0,x_fft.shape[sampling_axis])
    else:
        idxs = range(0,max_order)
    x_fft = np.take(x_fft, indices=idxs, axis=sampling_axis)
    return x_fft
