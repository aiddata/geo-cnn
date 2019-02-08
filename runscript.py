from __future__ import print_function, division

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import resnet


class RunCNN():

    def __init__(self, dataloaders, device, cat_names,
                 parallel=False, quiet=False, **kwargs):

        self.dataloaders = dataloaders
        self.device = device

        self.ncats = len(cat_names)

        self.parallel = parallel
        self.quiet = quiet

        self.kwargs = kwargs

        print("Initializing with kwargs: \n\t {}".format(kwargs))

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.state_dict = None

        resnet_args = {
            "pretrained":True,
            "n_input_channels": kwargs["n_input_channels"]
        }

        if kwargs["net"] == "resnet18":
            self.model = resnet.resnet18(**resnet_args)
        elif kwargs["net"] == "resnet34":
            self.model = resnet.resnet34(**resnet_args)
        elif kwargs["net"] == "resnet50":
            self.model = resnet.resnet50(**resnet_args)
        elif kwargs["net"] == "resnet101":
            self.model = resnet.resnet101(**resnet_args)
        elif kwargs["net"] == "resnet152":
            self.model = resnet.resnet152(**resnet_args)

        if self.model is None:
            raise Exception("Specified net not found ({})".format(kwargs["net"]))

        if self.kwargs["run_type"] == 2:
            for param in self.model.parameters():
                param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        # get existing number for input features
        # set new number for output features to number of categories being classified
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.ncats)

        loss_weights = torch.tensor(
            map(float, self.kwargs["loss_weights"])).cuda()

        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)


    def save(self, path):
        torch.save(self.model.state_dict(), path)


    def load(self, path):
        self.state_dict = torch.load(path)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()


    def export_to_device(self):

        if self.parallel and torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)


    def train(self, quiet=None):

        if quiet == None:
            quiet = self.quiet

        if self.kwargs["optim"] == "sgd":
            # Observe that only parameters of final layer
            # are being optimized as opposed to before.
            self.optimizer = optim.SGD(
                self.model.fc.parameters(),
                lr=self.kwargs["lr"],
                momentum=self.kwargs["momentum"])

        # Decay LR by a factor of `gamma` every `step_size` epochs
        # exp lr scheduler
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.kwargs["step_size"],
            gamma=self.kwargs["gamma"])

        self.export_to_device()

        best_acc, best_class_acc, time_elapsed = self._train(
            self.kwargs["n_epochs"], quiet)

        return best_acc, best_class_acc, time_elapsed


    def _train(self, num_epochs, quiet):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            if not quiet:
                print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_correct = 0
                running_count = 0

                class_correct = [0] * self.ncats
                class_count = [0] * self.ncats

                # iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)

                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)


                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    running_correct += torch.sum(preds == labels.data)
                    running_count += inputs.size(0)

                    for i in range(self.ncats):
                        label_indexes = (labels == i).nonzero().squeeze()
                        class_correct[i] += torch.sum(
                            preds[label_indexes] == labels[label_indexes]).item()
                        class_count[i] += len(label_indexes)


                epoch_loss = running_loss / running_count
                epoch_acc = running_correct.item() / running_count

                class_acc = [class_correct[i] / class_count[i] for i in range(self.ncats)]

                if not quiet:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_class_acc = class_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase == 'val':
                    for i in range(self.ncats):
                        print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
                            i, class_correct[i], class_count[i], class_acc[i]))


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.state_dict = best_model_wts

        return best_acc, best_class_acc, time_elapsed


    def test(self):

        epoch_loss, epoch_acc, class_acc, time_elapsed = self._test()

        return epoch_loss, epoch_acc, class_acc, time_elapsed


    def _test(self):

        phase = "test"

        since = time.time()

        self.export_to_device()

        self.model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        class_correct = [0] * self.ncats
        class_count = [0] * self.ncats

        # Iterate over data.
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)

            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)

            running_correct += torch.sum(preds == labels.data)
            running_count += inputs.size(0)

            for i in range(self.ncats):
                label_indexes = (labels == i).nonzero().squeeze()
                class_correct[i] += sum(preds[label_indexes] == labels[label_indexes]).item()
                class_count[i] += len(label_indexes)


        epoch_loss = running_loss / running_count
        epoch_acc = running_correct.item() / running_count

        class_acc = [class_correct[i] / class_count[i] for i in range(self.ncats)]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        for i in range(self.ncats):
            print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
                i, class_correct[i], class_count[i], class_acc[i]))


        time_elapsed = time.time() - since
        print('\nTesting complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        return epoch_loss, epoch_acc, class_acc, time_elapsed


    def predict(self):
        pass


    def _predict(self):
        since = time.time()

        model.eval()   # Set model to evaluate mode

        full_preds = []

        # iterate over data
        for inputs, _ in self.dataloaders[phase]:
            inputs = inputs.to(self.device)

            with torch.set_grad_enabled(0):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                full_preds += preds # need to test this

        time_elapsed = time.time() - since
        print('\nPrediction completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        return full_preds, time_elapsed







# -----------------------------------------------------------------------------







# def run(dataloaders, device, mode="train", quiet=False, **kwargs):

#     # print("\n{}:\n".format(run_types[kwargs["run_type"]]))
#     print(kwargs)

#     if kwargs["net"] == "resnet18":
#         model_x = resnet.resnet18(pretrained=True, n_input_channels=kwargs["n_input_channels"])
#     elif kwargs["net"] == "resnet34":
#         model_x = resnet.resnet34(pretrained=True, n_input_channels=kwargs["n_input_channels"])
#     elif kwargs["net"] == "resnet50":
#         model_x = resnet.resnet50(pretrained=True, n_input_channels=kwargs["n_input_channels"])
#     elif kwargs["net"] == "resnet101":
#         model_x = resnet.resnet101(pretrained=True, n_input_channels=kwargs["n_input_channels"])
#     elif kwargs["net"] == "resnet152":
#         model_x = resnet.resnet152(pretrained=True, n_input_channels=kwargs["n_input_channels"])
#     else:
#         raise Exception("net not found ({})".format(kwargs["net"]))


#     if mode == "train":

#         if kwargs["run_type"] == 2:
#             for param in model_x.parameters():
#                 param.requires_grad = False

#         # Parameters of newly constructed modules have requires_grad=True by default
#         num_ftrs = model_x.fc.in_features

#         model_x.fc = nn.Linear(num_ftrs, kwargs["ncats"])

#         loss_weights = torch.tensor(
#             map(float, kwargs["loss_weights"])).cuda()

#         criterion = nn.CrossEntropyLoss(weight=loss_weights)

#         if kwargs["optim"] == "sgd":
#             # Observe that only parameters of final layer are being optimized as opposed to before.
#             optimizer_x = optim.SGD(model_x.fc.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"])

#         # Decay LR by a factor of `gamma` every `step_size` epochs
#         exp_lr_scheduler = lr_scheduler.StepLR(optimizer_x, step_size=kwargs["step_size"], gamma=kwargs["gamma"])


#         # if torch.cuda.device_count() > 1:
#         #     print("Using", torch.cuda.device_count(), "GPUs")
#         #     model_x = nn.DataParallel(model_x)

#         model_x = model_x.to(device)

#         model_x, acc_x, class_x, time_x = train_model(
#             model_x, dataloaders, device, criterion, optimizer_x, exp_lr_scheduler,
#             num_epochs=kwargs["n_epochs"], quiet=quiet)


#     if mode == "test":
#         pass
#         # test_model(model_x, dataloaders, criterion, optimizer_x)


#     if mode == "predict":
#         pass
#         # predict_model(model_x, dataloaders)


#     return model_x, acc_x, class_x, time_x





# def train_model(model, dataloaders, device, criterion, optimizer, scheduler, num_epochs=25, quiet=True):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         if not quiet:
#             print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
#             print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_correct = 0
#             running_count = 0

#             class_correct = [0] * ncats
#             class_count = [0] * ncats

#             # iterate over data
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)

#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)


#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)

#                 running_correct += torch.sum(preds == labels.data)
#                 running_count += inputs.size(0)

#                 for i in range(ncats):
#                     label_indexes = (labels == i).nonzero().squeeze()
#                     class_correct[i] += torch.sum(preds[label_indexes] == labels[label_indexes]).item()
#                     class_count[i] += len(label_indexes)


#             epoch_loss = running_loss / running_count
#             epoch_acc = running_correct.item() / running_count

#             class_acc = [class_correct[i] / class_count[i] for i in range(ncats)]

#             if not quiet:
#                 print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                     phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_class_acc = class_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#             if phase == 'val':
#                 for i in range(ncats):
#                     print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
#                         i, class_correct[i], class_count[i], class_acc[i]))


#     time_elapsed = time.time() - since
#     print('\nTraining complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, best_acc, best_class_acc, time_elapsed


# def test_model(model, dataloaders, device, criterion, optimizer):
#     since = time.time()

#     model.eval()   # Set model to evaluate mode

#     running_loss = 0.0
#     running_correct = 0
#     running_count = 0

#     class_correct = [0] * ncats
#     class_count = [0] * ncats

#     # Iterate over data.
#     for inputs, labels in dataloaders[phase]:
#         inputs = inputs.to(device)

#         labels = labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         with torch.set_grad_enabled(0):
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)

#         # statistics
#         running_loss += loss.item() * inputs.size(0)

#         running_correct += torch.sum(preds == labels.data)
#         running_count += inputs.size(0)

#         for i in range(ncats):
#             label_indexes = (labels == i).nonzero().squeeze()
#             class_correct[i] += sum(preds[label_indexes] == labels[label_indexes]).item()
#             class_count[i] += len(label_indexes)


#     epoch_loss = running_loss / running_count
#     epoch_acc = running_correct.item() / running_count

#     class_acc = [class_correct[i] / class_count[i] for i in range(ncats)]

#     print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#         phase, epoch_loss, epoch_acc))

#     for i in range(ncats):
#         print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
#             i, class_correct[i], class_count[i], class_acc[i]))


#     time_elapsed = time.time() - since
#     print('\nTesting complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))

#     return epoch_loss, epoch_acc, class_acc, time_elapsed



# def predict_model(model, dataloaders, device):
#     since = time.time()

#     model.eval()   # Set model to evaluate mode

#     full_preds = []

#     # iterate over data
#     for inputs, _ in dataloaders[phase]:
#         inputs = inputs.to(device)

#         with torch.set_grad_enabled(0):
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             full_preds += preds # need to test this

#     time_elapsed = time.time() - since
#     print('\nPrediction completed in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))

#     # load best model weights
#     return full_preds, time_elapsed


