import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel

allowed_model_names = ['conv', 'rnn', 'lstm', 'gru', 'transformer', 'encoder_decoder', 'grouped_conv1d']


class Force_connex(nn.Module):
  def __init__(self,p):
    super().__init__()
    self.len_window = p.len_window
    self.low_threshold = p.low_threshold
    self.high_threshold = p.high_threshold

    self.avgpool = torch.nn.AvgPool1d(kernel_size=self.len_window,
          stride=1, padding=self.len_window//2, count_include_pad=False)

  def forward(self,x,threshold):
    y = x.unsqueeze(1)
    y = self.avgpool(y)
    y = torch.squeeze(y)
    surrounded_by_high = torch.gt(y,self.high_threshold)
    surrounded_by_low = torch.logical_not(torch.gt(y,self.low_threshold))
    decrease = torch.logical_and(torch.gt(x,threshold), surrounded_by_low)
    increase = torch.logical_and(torch.logical_not(torch.gt(x,threshold)), surrounded_by_high)
    x = torch.clip(x+increase.long()-decrease.long(), min=0, max=1)
    return x

class Conv1D(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.in_channels = len(p.signal_ids)
    self.use_maxpool = p.use_maxpool
    self.use_avgpool = p.use_avgpool
    assert len(self.ks) == 3

    self.conv1 = nn.Conv1d(self.in_channels, 16, 5)
    self.conv2 = nn.Conv1d(16, 32, 5)
    self.conv3 = nn.Conv1d(32, 64, 3)

    self.maxpool = nn.MaxPool1d(10)
    self.avgpool = nn.AdaptiveAvgPool1d(100)
    self.dropout = nn.Dropout(p.dropout_p)

    self.fc = nn.Linear(100*64, self.seq_length*self.conv_output_dim)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout(x)
    if self.use_maxpool:
      x = self.maxpool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)
    if self.use_maxpool:
      x = self.maxpool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout(x)
    if self.use_avgpool:
      x = self.avgpool(x)

    x = self.flatten(x)
    x = self.fc(x)

    return x


class GroupedConv1D(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.seq_length = p.seq_length
    self.in_channels = len(p.signal_ids)

    self.conv1 = nn.Conv1d(
      in_channels=self.seq_length, 
      out_channels=self.seq_length*4, 
      kernel_size=50, 
      groups=90
    )
    self.conv2 = nn.Conv1d(
      in_channels=self.seq_length*4, 
      out_channels=self.seq_length*8, 
      kernel_size=20, 
      groups=90
    )
    self.conv3 = nn.Conv1d(
      in_channels=self.seq_length*8, 
      out_channels=self.seq_length*16, 
      kernel_size=3, 
      groups=90
    )
    self.conv4 = nn.Conv1d(
      90*16,
      90,
      kernel_size=1,
      groups=90
    )

    self.dropout = nn.Dropout(p.dropout_p)
    self.fc = nn.Linear(30, 1)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv4(x)
    x = self.fc(x)

    return torch.sigmoid(x)


class Conv2D(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.in_channels = len(p.signal_ids)

    self.conv1 = nn.Conv2d(self.in_channels, 4, kernel_size=(1,50))
    self.conv2 = nn.Conv2d(4, 8, kernel_size=(1,20))
    self.conv3 = nn.Conv2d(8, 16, kernel_size=(1,3))
    self.conv4 = nn.Conv2d(16, 1, kernel_size=(1,1))

    self.maxpool = nn.MaxPool2d(kernel_size=(1,2))
    self.avgpool = nn.AdaptiveAvgPool2d(100)
    self.dropout = nn.Dropout(p.dropout_p)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv4(x)
    x = self.relu(x)

    # for good RNN integration, x should be (batch, seq_len, conv_output_dim)
    x = x.squeeze(1)

    return x


class GroupedConv2D(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.seq_length = p.seq_length
    self.in_channels = len(p.signal_ids)
    self.n_groups = p.n_groups

    self.conv1 = nn.Conv2d(
      self.in_channels,
      self.in_channels*self.n_groups,
      kernel_size=(1,50),
      groups=self.in_channels
    )
    self.conv2 = nn.Conv2d(
      self.in_channels*self.n_groups,
      self.in_channels*self.n_groups*2,
      kernel_size=(1,20),
      groups=self.in_channels
    )
    self.conv3 = nn.Conv2d(
      self.in_channels*self.n_groups*2,
      self.in_channels*self.n_groups*4,
      kernel_size=(1,3),
      groups=self.in_channels
    )
    self.conv4 = nn.Conv2d(
      self.in_channels*self.n_groups*4,
      self.in_channels,
      kernel_size=(1,1),
      groups=self.in_channels
    )

    self.dropout = nn.Dropout(p.dropout_p)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv4(x)
    x = self.relu(x)

    x = x.permute(0, 2, 3, 1)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])

    # for good RNN integration, x should be (batch, seq_len, conv_output_dim)
    # x = x.squeeze(1)

    return x


class RNN(nn.Module):
  def __init__(self, p):

    super().__init__()

    self.bidirectional = p.bidirectional

    self.rnn = nn.RNN(input_size=p.input_dim,
                      hidden_size=p.hidden_dim,
                      num_layers=p.n_layers,
                      bidirectional=p.bidirectional,
                      dropout=p.dropout_p,
                      batch_first=True)

    fc_input_dim = 2*p.hidden_dim if self.bidirectional else p.hidden_dim
    self.fc = nn.Linear(in_features=fc_input_dim, out_features=1)
    self.dropout = nn.Dropout(p.dropout_p)

  def forward(self, x):

    x, hidden = self.rnn(x)
    x = self.fc(x)
    x = x.squeeze()

    return torch.sigmoid(x)


class LSTM(nn.Module):
  def __init__(self, p):

    super().__init__()

    self.bidirectional = p.bidirectional

    self.rnn = nn.LSTM(input_size=p.input_dim,
                      hidden_size=p.hidden_dim,
                      num_layers=p.n_layers,
                      bidirectional=p.bidirectional,
                      dropout=p.dropout_p,
                      batch_first=True)

    conv_input_dim = 2*p.hidden_dim if self.bidirectional else p.hidden_dim
    self.conv = nn.Conv2d(1, 1, kernel_size=(1,conv_input_dim))
    self.fc = nn.Linear(in_features=conv_input_dim, out_features=1)
    self.dropout = nn.Dropout(p.dropout_p)
    self.last_layer = p.last_layer

  def forward(self, x):

    x, (hidden, cell) = self.rnn(x)

    if self.last_layer == 'fc':
      x = self.fc(x)
    elif self.last_layer == 'conv':
      x = x.unsqueeze(1)
      x = self.conv(x)
    else:
      ValueError(f'{self.last_layer} not supported yet')
    x = x.squeeze()

    return torch.sigmoid(x)


class BERT(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.bert_config = transformers.BertConfig(
      vocab_size=1,
      hidden_size=p.input_dim,
      num_hidden_layers=p.n_layers,
      num_attention_heads=p.n_heads,
      intermediate_size=p.ffn_dim,
      hidden_dropout_prob=p.dropout_p,
      attention_probs_dropout_prob=p.dropout_p,
      max_position_embeddings=p.seq_length,
      position_embedding_type='absolute'
    )

    self.bert = BertModel(self.bert_config)
    self.conv = nn.Conv2d(1, 1, kernel_size=(1,p.input_dim))
    self.fc = nn.Linear(in_features=p.input_dim, out_features=1)
    self.relu = nn.ReLU()
    self.last_layer = p.last_layer

  def forward(self, x):

    x = self.bert(inputs_embeds=x)
    x = x['last_hidden_state']
    if self.last_layer == 'fc':
      x = self.fc(x)
    elif self.last_layer == 'conv':
      x = x.unsqueeze(1)
      x = self.conv(x)
    else:
      ValueError(f'{self.last_layer} not supported yet')
    x = x.squeeze()

    return torch.sigmoid(x)


class EncoderDecoder(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.model = p.model
    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.relu = nn.ReLU()

    if p.encoder == 'conv2d':
      self.encoder = Conv2D(p)
    elif p.encoder == 'grouped_conv2d':
      self.encoder = GroupedConv2D(p)
      p.conv_output_dim *= len(p.signal_ids)
    else:
      raise ValueError(f'{p.encoder} not supported yet')

    if p.decoder == 'lstm':
      p.input_dim = p.conv_output_dim
      self.decoder = LSTM(p)
    elif p.decoder == 'transformer':
      p.input_dim = p.conv_output_dim
      self.decoder = BERT(p)
    elif p.decoder != 'conv':
      raise ValueError(f'{p.model} not supported yet')

  def forward(self, x):

    x = self.encoder(x)
    x = self.relu(x)
    x = self.decoder(x)

    return x


def create_model(p):

    model = None
    if p.model not in allowed_model_names:
      raise ValueError(f'Model {p.model} not recognized')
    else:
      if p.model == 'encoder_decoder':
        model = EncoderDecoder(p)
        print(f'{p.model} was created: {p.encoder}+{p.decoder}\n')
      else:
        if p.model == 'rnn':
          p.input_dim = p.input_dim * len(p.signal_ids)
          model = RNN(p)
        elif p.model == 'lstm':
          p.input_dim = p.input_dim * len(p.signal_ids)
          model = LSTM(p)
        elif p.model == 'transformer':
          p.input_dim = p.input_dim * len(p.signal_ids)
          model = BERT(p)
        elif p.model == 'grouped_conv1d':
          model = GroupedConv1D(p)
        print(f'{p.model} was created\n')

    return model
