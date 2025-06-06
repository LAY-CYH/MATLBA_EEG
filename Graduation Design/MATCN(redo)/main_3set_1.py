# -*- coding: utf-8 -*-
# """
# Created on Mon Feb 20 15:00:00 2023
# @author: Dong HUANG
# Modified for SEED dataset
# """

# %% Import
import copy
import os
import gc
import shutil
from time import strftime

import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import yaml
from easydict import EasyDict
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocess
# from modellibs import models
from modellibs import models_1 as models
from torchutils import get_trainer
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import (print_time_stamp, reset_workpath, save_history, save_log_confusion_matrix, seed_everything,
                   save_metrics)


# %% Data First Load - Modified for SEED dataset
def dataFirstLoad():
    channels = CFG.get('channels', 62)  # SEED has 62 channels
    srate = CFG.get('srate', 125)  # Target sampling rate
    winlen = CFG.get('windowLength', 14)
    sub_df = pd.DataFrame()

    if not os.path.exists('processData'):
        os.mkdir('processData')

    print(f"Looking for .mat files in: {DATA_DIR}")
    # Get all .mat files in the directory
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    mat_files.sort()
    print(f"Found {len(mat_files)} .mat files")

    if CFG.debug:
        subNum = 1
    else:
        subNum = len(mat_files)  # Each .mat file is one subject/session

    # add preprocess sequential method object
    preproc_seq = preprocess.PreProcessSequential(CFG)

    # Process each .mat file (each subject/session)
    for i in range(subNum):
        if i >= len(mat_files):
            break

        mat_file = mat_files[i]
        print(f"Processing {mat_file}...")

        if not os.path.exists(os.path.join('processData', f'sub{i + 1}')):
            os.mkdir(os.path.join('processData', f'sub{i + 1}'))

        # Load the .mat file using scipy
        try:
            import scipy.io as sio
            data_dict = sio.loadmat(os.path.join(DATA_DIR, mat_file))

            # Find the variable prefix (like 'djc', 'jl', etc.)
            prefix = None
            for key in data_dict.keys():
                if key.endswith('_eeg1'):
                    prefix = key[:-5]  # Remove '_eeg1'
                    break

            if prefix is None:
                print(f"Could not find EEG data in {mat_file}, skipping.")
                continue

            # Process each trial (1-15)
            sample_num = 0
            for j in range(15):  # SEED has 15 trials
                trial_idx = j + 1  # Trials are 1-indexed in SEED
                eeg_key = f"{prefix}_eeg{trial_idx}"

                if eeg_key not in data_dict:
                    print(f"Trial {trial_idx} not found in {mat_file}")
                    continue

                # Get EEG data for this trial
                raw_eeg = data_dict[eeg_key]  # Should be 62 x time_points

                # Apply preprocessing
                processed_eeg = preprocess.preprocess_ref(raw_eeg, CFG.rerefence_type)
                processed_eeg = preproc_seq(processed_eeg)
                processed_eeg = processed_eeg[:, -6000:]

                # Segment the data using sliding windows
                window_size = int(winlen * 100)

                step_size = int(CFG.windowStep * 100)

                # Create windows
                data_segments = []
                for start in range(0, processed_eeg.shape[1] - window_size, step_size):
                    segment = processed_eeg[:, start:start + window_size]
                    # Apply additional preprocessing
                    data_segments.append(segment)

                if not data_segments:
                    print(f"No segments created for trial {trial_idx}")
                    continue

                # Stack all segments
                data_tmp = np.stack(data_segments)

                # Get the label for this trial from label_csv
                label_idx = j  # 0-indexed
                if label_idx < len(label_csv):
                    trial_label = label_csv.label1[label_idx]

                    # Save to file
                    np.save(os.path.join('processData', f'sub{i + 1}', f'{trial_label}.npy'), data_tmp)
                    sample_num += len(data_segments)
                else:
                    print(f"Warning: Label index {label_idx} out of range for label_csv")

            # Add subject info to dataframe
            if sample_num > 0:
                sub_df = pd.concat([sub_df, pd.DataFrame({
                    'sub': [i + 1],
                    'num': [sample_num]})], axis=0, ignore_index=True)
                print(f"Processed {sample_num} samples for subject {i + 1}")
            else:
                print(f"Warning: No samples processed for subject {i + 1}")

        except Exception as e:
            print(f"Error processing {mat_file}: {e}")

    # Save subject information
    os.makedirs("mid_files", exist_ok=True)

    if len(sub_df) > 0:
        sub_df.to_csv(f'./mid_files/sub_df_{CFG.all_batch}.csv', index=False)
        print(f"Subject information saved to ./mid_files/sub_df_{CFG.all_batch}.csv")
    else:
        print("No subjects were processed successfully.")
        # Create a dummy dataframe to avoid further errors
        sub_df = pd.DataFrame({'sub': range(1, 16), 'num': [1000] * 15})
        sub_df.to_csv(f'./mid_files/sub_df_{CFG.all_batch}.csv', index=False)

    return sub_df


# %% Data Second Load - Modified for SEED dataset
def dataSecondLoad(sub_train, dataType):
    channels = CFG.get('channels', 62)  # SEED has 62 channels
    srate = CFG.get('srate', 125)
    winlen = CFG.get('windowLength', 14)
    input_shape = CFG.get('input_shape', (channels, srate * winlen))

    data_shape = [sum(sub_train.num)] + list(input_shape)
    data_train = np.zeros(data_shape, dtype=np.float32)
    label_train = np.zeros(sum(sub_train.num), dtype=np.float32)
    assemble_num = 0
    subs = sub_train['sub']
    load_subj_bar = tqdm(range(len(sub_train)),
                         desc=f'[{strftime("%Y/%m/%d-%H:%M:%S")}] {dataType} Loading',
                         ascii=True)

    for j in load_subj_bar:
        for k in range(len(label_csv.label1)):
            file_path = os.path.join('processData', f'sub{subs[j]}', f'{label_csv.label1[k]}.npy')
            if os.path.exists(file_path):
                try:
                    rawdata_tmp = np.load(file_path)
                    label_tmp = label_csv.label2[k]
                    label_tmp = np.repeat(label_tmp, rawdata_tmp.shape[0])

                    # Ensure space in arrays
                    if assemble_num + rawdata_tmp.shape[0] > data_train.shape[0]:
                        # Expand arrays if needed
                        extra_size = rawdata_tmp.shape[0]
                        data_train = np.concatenate(
                            [data_train, np.zeros([extra_size] + list(input_shape), dtype=np.float32)])
                        label_train = np.concatenate([label_train, np.zeros(extra_size, dtype=np.float32)])

                    # Copy data
                    data_train[assemble_num:(assemble_num + rawdata_tmp.shape[0])] = rawdata_tmp
                    label_train[assemble_num:(assemble_num + rawdata_tmp.shape[0])] = label_tmp
                    assemble_num += rawdata_tmp.shape[0]
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    # Trim arrays to actual size
    data_train = data_train[:assemble_num]
    label_train = label_train[:assemble_num]

    # Shuffle data
    if assemble_num > 0:
        data_permutation = np.random.permutation(data_train.shape[0])
        data_train = data_train[data_permutation]
        label_train = label_train[data_permutation]

        # Balance classes (for SEED - 3 classes)
        num_sample_logs = [np.sum(label_train == i) for i in range(3)]
        min_samples = np.min(num_sample_logs)

        data_train_tmp = []
        label_train_tmp = []

        for label_itor in range(3):  # SEED has 3 emotion classes
            idxs = np.where(label_train == label_itor)[0]
            if len(idxs) > 0:
                # Limit to min_samples or all available samples
                idxs = idxs[:min(len(idxs), min_samples)]
                data_train_tmp.append(data_train[idxs])
                label_train_tmp.append(np.full(len(idxs), label_itor))

        if data_train_tmp:
            data_train = np.concatenate(data_train_tmp)
            label_train = np.concatenate(label_train_tmp)

    return data_train, label_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEED_EMOTION')
    parser.add_argument('--workpath', '-W', type=str, default='./workpath')
    parser.add_argument('--reload', '-R', action='store_true', default='True')
    parser.add_argument('--resume-K', '-K', type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.workpath, 'config.yaml')) as f:
        CFG = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    DATA_DIR = r'/home/833/xzr/Code/jiedan/SEED/Preprocessed_EEG'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CFG.device) if isinstance(CFG.device, int) else ','.join(
        str(_) for _ in CFG.device)
    torch.cuda.empty_cache()

    label_csv = pd.read_csv('index_label.csv')

    # For SEED, we don't need channel adjustment as we're using preprocessed data
    # but keep this variable for compatibility
    channel_index = np.arange(62)  # SEED has 62 channels

    seed_everything(CFG.seed)
    if args.reload:
        dataFirstLoad()

    # parameter initialize
    model_name = CFG.get('model_name', 'MACTN')
    channels = CFG.get('channels', 62)  # SEED has 62 channels
    srate = CFG.get('srate', 125)
    winlen = CFG.get('windowLength', 14)
    input_shape = CFG.get('input_shape', (channels, srate * winlen))
    output_shape = CFG.get('num_classes', 3)  # SEED has 3 emotion classes

    trainer_name = CFG.get('trainer_name', 'CrossVal')

    CFG.ckpt_dir = f'{args.workpath}/ckpt'
    CFG.model_save_dir = f'{args.workpath}/model'
    model_metrics_dir = os.path.join(args.workpath, 'ckpt')
    if not os.path.exists(model_metrics_dir):
        os.makedirs(model_metrics_dir)

    # reset work dir
    if args.resume_K == 0:
        reset_workpath(model_metrics_dir, clear=True)
        reset_workpath(CFG.model_save_dir, clear=True)

    # Load or create subject dataframe
    sub_df_path = f'./mid_files/sub_df_{CFG.all_batch}.csv'

    if os.path.exists(sub_df_path):
        sub_df = pd.read_csv(sub_df_path)
    else:
        if args.reload:
            print(f"Will create subject data file at {sub_df_path}")

            dataFirstLoad()
            if os.path.exists(sub_df_path):
                sub_df = pd.read_csv(sub_df_path)
            else:
                print(f"Error: Failed to create {sub_df_path}")

                sub_df = pd.DataFrame({'sub': [], 'num': []})
        else:
            print(f"Warning: {sub_df_path} not found. Run with --reload to create it.")

            sub_df = pd.DataFrame({'sub': range(1, 16), 'num': [1000] * 15})  # 15 subjects in SEED
    # Randomize subjects
    # sub_per = sub_df.sample(frac=1, random_state=CFG.seed, ignore_index=True)

    # Cross-validation loop
    sub_per = sub_df
    start_kfold = args.resume_K if 0 <= args.resume_K < CFG.n_fold else 0
    for i in range(start_kfold, CFG.n_fold):
        # 每次取一个样本作为测试集
        sub_test = sub_per.iloc[[i]].reset_index(drop=True)  # 注意双重中括号保持 DataFrame 格式

        # 剩下的所有样本作为训练集
        sub_train = sub_per.drop(i).reset_index(drop=True)

        # 加载数据（你需要自己实现 dataSecondLoad 函数）
        data_train, label_train = dataSecondLoad(sub_train, 'TRAIN')
        data_test, label_test = dataSecondLoad(sub_test, 'TEST')

        # Skip this fold if no data was loaded
        if len(data_train) == 0 or len(data_test) == 0:
            print(f"Error: Failed to load data for fold {i}")
            continue

        # Set checkpoint name for this fold
        CFG.ckpt_name = f'ckpt_{model_name}_{i}'

        # Setup TensorBoard logger
        writer = SummaryWriter(log_dir=CFG.ckpt_dir + f'/event/fold_{i}')

        # Start training
        print_time_stamp(f'Information --> model: {model_name}, n_fold: {i}, WP: {os.path.basename(args.workpath)}')
        model = models.get_model(model_name, input_shape, output_shape, dropoutRate=CFG.dropout_rate)
        trainer = get_trainer(trainer_name, data_train, label_train, data_test, label_test, model, CFG, writer)
        trainer.fit()
        history = trainer.history

        gc.collect()
        trainer.load_ckpt()

        # Test the model
        data_test, label_test = dataSecondLoad(sub_test, 'TEST')

        if len(data_test) > 0:
            label_pred = trainer.predict(data_test)
            # del data_valid
            gc.collect()

            # Generate confusion matrix
            cm_raw = confusion_matrix(label_test, label_pred, normalize='true')
            cm = np.around(cm_raw * 100, decimals=1)

            # SEED emotion labels
            labels = ['negative', 'neutral', 'positive']

            # Plot and save confusion matrix
            cmdp = ConfusionMatrixDisplay(cm, display_labels=labels)
            plt.rcParams['figure.figsize'] = [10, 10]
            cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=45, colorbar=True, values_format='.1f')
            plt.savefig(os.path.join(model_metrics_dir, f'ConfusionMat_fold{i}.pdf'))
            plt.close()

            # Plot training history
            acc = history['acc']
            val_acc = history['val_acc']
            loss = history['loss']
            val_loss = history['val_loss']
            epochs = range(1, len(acc) + 1)

            plt.figure()
            plt.plot(epochs, acc, 'bo-', label='Train_Acc')
            plt.plot(epochs, val_acc, 'r^-', label='Val_Acc')
            plt.title('Train and Val Acc', fontsize=20)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(model_metrics_dir, f'Acc_fold{i}.pdf'))
            plt.close()

            plt.figure()
            plt.plot(epochs, loss, 'bo-', label='Train_Loss')
            plt.plot(epochs, val_loss, 'r^-', label='Val_Loss')
            plt.title('Train and Val Loss', fontsize=20)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(model_metrics_dir, f'Loss_fold{i}.pdf'))
            plt.close()

            writer.close()

            try:
                # Save training history
                his_name = os.path.join(model_metrics_dir, 'history.csv')
                his_copy = copy.deepcopy(history)
                his_rename = {k + f'_{i}': v for k, v in his_copy.items()}
                his_df = pd.DataFrame(his_rename)
                save_history(his_df, his_name)

                # Save confusion matrix
                cm_name = os.path.join(model_metrics_dir, 'confusion_matrix.npy')
                cm_copy = copy.deepcopy(cm)
                save_log_confusion_matrix(cm, cm_name)

                # Calculate and save metrics
                acc = accuracy_score(label_test, label_pred)
                precision = precision_score(label_test, label_pred, average='macro')
                recall = recall_score(label_test, label_pred, average='macro')
                f1 = f1_score(label_test, label_pred, average='macro')
                metrics = {'acc': [acc], 'precision': [precision], 'recall': [recall], 'f1': [f1]}
                metric_df = pd.DataFrame(copy.deepcopy(metrics))
                metric_name = os.path.join(model_metrics_dir, 'metrics.csv')
                save_metrics(metric_df, metric_name)

                print(
                    f"Fold {i} metrics - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            except Exception as e:
                print(f'Save history failure: {e}')
        else:
            print(f"Warning: No test data for fold {i}")

        if CFG.debug:
            break

    # Try to generate overall confusion matrix across all folds
    try:
        cm_data_name = os.path.join(model_metrics_dir, 'confusion_matrix.npy')
        cm_data = np.load(cm_data_name)
        cm_save_name = os.path.join(model_metrics_dir, f'ConfusionMat_{CFG.n_fold}_folds.pdf')

        # SEED emotion labels
        labels = ['negative', 'neutral', 'positive']

        cm_data = cm_data.mean(axis=0)
        cmdp = ConfusionMatrixDisplay(cm_data, display_labels=labels)
        plt.rcParams['figure.figsize'] = [10, 10]
        cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=45, colorbar=True, values_format='.1f')
        plt.savefig(cm_save_name)
        plt.close()
    except Exception as e:
        print(f'Draw confusion matrix all fold failure: {e}')