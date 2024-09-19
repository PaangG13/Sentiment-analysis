# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform
import matplotlib.pyplot as plt
from data_sst2 import DataPrecessForSentence
from utils import train, validate
from model import BertModel,Bert_Blend_CNN
import random


def model_train_validate_test(train_df, dev_df, test_df, target_dir,
                              max_seq_len=50,
                              epochs=40,
                              batch_size=32,
                              lr=2e-05,
                              patience=1,
                              max_grad_norm=10.0,
                              if_save_model=True,
                              checkpoint=None):
    """
    Parameters
    ----------
    train_df : pandas dataframe of train set.
    dev_df : pandas dataframe of dev set.
    test_df : pandas dataframe of test set.
    target_dir : the path where you want to save model.
    max_seq_len: the max truncated length.
    epochs : the default is 3.
    batch_size : the default is 32.
    lr : learning rate, the default is 2e-05.
    patience : the default is 1.
    max_grad_norm : the default is 10.0.
    if_save_model: if save the trained model to the target dir.
    checkpoint : the default is None.

    """

    bertmodel = Bert_Blend_CNN(requires_grad=True)
    tokenizer = bertmodel.tokenizer

    print(20 * "-", " Preparing for training ", 20 * "-")
    # Path to save the model, create a folder if not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading --------------------------------------#

    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=max_seq_len)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- --------------#

    print("\t* Building model...")
    device = torch.device("cuda")
    model = bertmodel.to(device)

    # -------------------- Preparation for training  -------------------------#

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # When the monitored value is not improving, the network performance could be improved by reducing the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []
    valid_f1 = []
    test_accuracies = []
    test_aucs = []
    test_f1 = []
    test_precision =[]
    test_recall = []
    best_f1 = 0.0
    best_epoch = 0

    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        train_accuracy = checkpoint["train_accuracy"]
        valid_losses = checkpoint["valid_losses"]
        valid_accuracy = checkpoint["valid_accuracy"]
        valid_auc = checkpoint["valid_auc"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, precision, recall, f1, auc, _, = validate(model, dev_loader)
    print(
        "\n* Validation before training:\nloss {:.4f}, accuracy: {:.2f}%, Precison: {:.2f}%\nRecall: {:.2f}%, F1: {:.2f}%, auc: {:.4f}".format(
            valid_loss, (valid_accuracy * 100), (precision * 100), (recall * 100), (f1 * 100), auc))

    # -------------------- Training epochs -----------------------------------#

    print("\n", 20 * "-", "Training bert model on device: {}".format(device), 20 * "-")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        if epoch_accuracy is None: 
          epoch_accuracy = 0
        train_accuracies.append(epoch_accuracy)
        
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%\n".format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, precision, recall, f1, epoch_auc, _, = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_accuracy)
        valid_f1.append(f1)
        if epoch_accuracy is None: 
          epoch_accuracy = 0
        valid_aucs.append(epoch_auc)
        print(
            "-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.2f}%, Precison:{:.2f}%\n          Recall:{:.2f}%, F1:{:.2f}%,    auc: {:.4f}\n"
                .format(epoch_time, epoch_loss, (epoch_accuracy * 100), (precision * 100), (recall * 100), (f1 * 100),
                        epoch_auc))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        ## scheduler.step()

        # run model on test set and save the prediction result to csv
        print("* Test for epoch {}:".format(epoch))
        _, _, test_accuracy, precision, recall, f1, test_auc, all_prob = validate(model, test_loader)
        test_accuracies.append(test_accuracy)
        test_precision.append(precision)
        test_recall.append(recall)
        test_aucs.append(test_auc)
        test_f1.append(f1)
        print(
            "Test Accuracy: {:.2f}%, Test Precison:{:.2f}%,\nTest Recall:{:.2f}%,    Test F1:{:.2f}%\tTest AUC:{:.4f}\n".format(
                (test_accuracy * 100), (precision * 100), (recall * 100), (f1 * 100), test_auc))
        print("-" * 50)
        test_prediction = pd.DataFrame({'prob_1': all_prob})
        test_prediction['prob_0'] = 1 - test_prediction['prob_1']
        test_prediction['Prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
        test_prediction['Label'] = test_df['similarity'].values
        test_prediction['Sentence'] = test_df["s1"].values
        pt = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
        sentences = test_df["s1"].values
        labels = test_df['similarity'].values
        if(f1>best_f1):
            best_epoch = epoch-1
        # Early stopping on validation accuracy.
        if f1 < best_score:
            patience_counter += 1
        else:
            best_score = f1
            patience_counter = 0
            if (if_save_model):
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_score": best_score,
                            "epochs_count": epochs_count,
                            "train_losses": train_losses,
                            "train_accuracy": train_accuracies,
                            "valid_losses": valid_losses,
                            "valid_accuracy": valid_accuracies,
                            "valid_f1": f1,
                            "valid_auc": valid_aucs
                            },
                           os.path.join(target_dir, "best.pth.tar"))
                print("save model succesfully!\n")
    # plot
    '''
    plt.plot(epochs_count, train_losses, color='r', label='train_loss')
    plt.plot(epochs_count, valid_losses, color='b', label='train_loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    '''
    plt.figure(1)
    plt.plot(epochs_count, train_losses, color='orange', label='train_loss')
    plt.plot(epochs_count, valid_losses, label='train_loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(epochs_count, valid_accuracies, label='valid_accuracy')
    plt.plot(epochs_count, test_accuracies, color='orange', label='test_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(epochs_count, valid_f1, label='valid_f1')
    plt.plot(epochs_count, test_f1, color='orange', label='test_f1')
    plt.xlabel('epochs')
    plt.ylabel('f1')
    # plt.title('Curve of Temperature Change with Time')
    plt.legend()
    plt.show()
    
    print(epochs_count)
    print(train_losses)
    print(valid_losses)
    print(valid_accuracies)
    print(test_accuracies)
    print(valid_f1)
    print(test_f1)
    print("Best model:")
    print(
        "Accuracy: {:.2f}%, \tPrecison:{:.2f}%,\nTest Recall:{:.2f}%,\tTest F1:{:.2f}%\tTest AUC:{:.4f}\n".format(
            (test_accuracies[best_epoch] * 100), (test_precision[best_epoch] * 100), (test_recall[best_epoch] * 100), (test_f1[best_epoch] * 100), test_aucs[best_epoch]))
    print("Demo:")
    for i in range(1,11):
        ii=random.randint(1, 800)
        print("Test",i)
        print("Sentences:",sentences[ii])
        print("Labels:",labels[ii],"Prediction:",pt[ii])
    test_prediction = test_prediction[['Sentence','Label','prob_0', 'prob_1', 'Prediction']]
    test_prediction.to_csv(os.path.join(target_dir, "test_prediction.csv"), index=False)



if __name__ == "__main__":
    data_path = "./data/"
    train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    target_dir = "./output/Bert/"
    model_train_validate_test(train_df, dev_df, test_df, target_dir)
