import torch
import torch.nn as nn


class RnnEncoder(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, cell_type='GRU', num_layers=1, device='cpu', dropout=0, bidirectional=True, mask_mode="all"):
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device
        self.mask_mode = mask_mode

        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)

        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x, mask=None):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)

        # Compute RNN output for all mask modes
        out, _ = self.rnn(x.to(self.device), past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]

        if mask is None:
            mask = self.mask_mode
        

        # Apply masking based on the mask mode
        if mask == "all":    
            encodings = self.nn(out[-1].squeeze(0))
        elif mask == "mask_last":
            mask = x.new_full((x.size(1), x.size(0)), True, dtype=torch.bool)
            mask[:, -1] = False
            out = out.transpose(0, 1)  # Transpose out to match mask dimensions
            out[~mask] = 0
            out = out.transpose(0, 1)  # Transpose back to original dimensions
            encodings = self.nn(out[-1].squeeze(0))
        else:
            raise ValueError('Mask mode not defined, must be one of the following {all, mask_last}')

        return encodings



class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = torch.nn.BatchNorm1d(self.input_size)
        self.nn = torch.nn.Linear(self.input_size, self.output_size)
        torch.nn.init.xavier_uniform_(self.nn.weight)

    def forward(self, x):
        x = self.normalize(x)
        logits = self.nn(x)
        return logits


class WFClassifier(torch.nn.Module): #simple linear classifier usable for classification tasks
    def __init__(self, encoding_size, output_size):
        super(WFClassifier, self).__init__()
        self.encoding_size = encoding_size
        self.output_size = output_size
        self.classifier = nn.Linear(self.encoding_size, output_size)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        c = self.classifier(x)
        return c


class E2EStateClassifier(torch.nn.Module):
    """ MANUALLY inserted
    End-to-End State Classifier module.

    Args:
        hidden_size (int): The size of the hidden state in the RNN.
        in_channel (int): The number of input channels.
        encoding_size (int): The size of the encoding layer.
        output_size (int): The size of the output layer.
        cell_type (str, optional): The type of RNN cell to use. Defaults to 'GRU'.
        num_layers (int, optional): The number of layers in the RNN. Defaults to 1.
        dropout (float, optional): The dropout probability. Defaults to 0.
        bidirectional (bool, optional): Whether to use bidirectional RNN. Defaults to True.
        device (str, optional): The device to run the module on. Defaults to 'cpu'.
    """

    def __init__(self, hidden_size, in_channel, encoding_size, output_size, cell_type='GRU', num_layers=1, dropout=0,
                 bidirectional=True, device='cpu'):
        super(E2EStateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.device = device

        self.fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.encoding_size, self.output_size)).to(self.device)
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        """
        Forward pass of the E2EStateClassifier module.

        Args:
            x (torch.Tensor): The input tensor of shape (seq_len, batch_size, in_channel).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.fc(out[-1].squeeze(0))
        return self.nn(encodings)


class MimicEncoder(torch.nn.Module):
    """Manually added
    A class representing the MimicEncoder module.

    Args:
        input_size (int): The size of the input tensor.
        in_channel (int): The number of input channels.
        encoding_size (int): The size of the output encoding.

    Attributes:
        input_size (int): The size of the input tensor.
        in_channel (int): The number of input channels.
        encoding_size (int): The size of the output encoding.
        nn (torch.nn.Sequential): The neural network layers.

    """

    def __init__(self, input_size, in_channel, encoding_size):
        super(MimicEncoder, self).__init__()
        self.input_size = input_size
        self.in_channel = in_channel
        self.encoding_size = encoding_size

        self.nn = torch.nn.Sequential(torch.nn.Linear(input_size, 64),
                                      torch.nn.Dropout(),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, encoding_size))

    def forward(self, x):
        """
        Forward pass of the MimicEncoder module.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The encoded tensor.
        """
        x = torch.mean(x, dim=1)
        encodings = self.nn(x)
        return encodings


class WFEncoder(nn.Module):
    def __init__(self, encoding_size, classify=False, n_classes=None):
        """
        Encoder module.

        Args:
            encoding_size (int): The size of the encoding vector.
            classify (bool, optional): Whether to include a classifier layer. Defaults to False.
            n_classes (int, optional): The number of output classes for the encoder. Required if classify is True.
        Raises:
            ValueError: If n_classes is not specified when classify is True.
        """
        super(WFEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.n_classes = n_classes
        self.classify = classify
        self.classifier = None
        if self.classify:
            if self.n_classes is None:
                raise ValueError('Need to specify the number of output classes for the encoder')
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.encoding_size, self.n_classes)
                )
                nn.init.xavier_uniform_(self.classifier[1].weight)

        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(79872, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Linear(2048, self.encoding_size)
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        encoding = self.fc(x)
        if self.classify:
            c = self.classifier(encoding)
            return c
        else:
            return encoding