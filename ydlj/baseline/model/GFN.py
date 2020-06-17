import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim

        self.sp_linear = nn.Linear(self.input_dim, 1)
        self.start_linear = nn.Linear(self.input_dim, 1)
        self.end_linear = nn.Linear(self.input_dim, 1)

        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        query_mapping = batch['query_mapping']  
        context_mask = batch['context_mask']  
        all_mapping = batch['all_mapping']  


        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)

        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2) 

        sp_state = sp_state.max(1)[0]

        sp_logits = self.sp_linear(sp_state)

        type_state = torch.max(input_state, dim=1)[0]
        type_logits = self.type_linear(type_state)

        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:   
            outer = outer - 1e30 * query_mapping[:, :, None]    

        
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]

        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position
class BertSupportNet(nn.Module):
    """
    joint train bert and graph fusion net
    """

    def __init__(self, config, encoder):
        super(BertSupportNet, self).__init__()
       
        self.encoder = encoder
        self.graph_fusion_net = SupportNet(config)

    def forward(self, batch, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']

        all_doc_encoder_layers = self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=doc_mask)[0]
        batch['context_encoding'] = all_doc_encoder_layers

        return self.graph_fusion_net(batch)


class SupportNet(nn.Module):
    """
    Packing Query Version
    """

    def __init__(self, config):
        super(SupportNet, self).__init__()
        self.config = config  
        
        self.max_query_length = 50
        self.prediction_layer = SimplePredictionLayer(config)

    def forward(self, batch, debug=False):
        context_encoding = batch['context_encoding']
        predictions = self.prediction_layer(batch, context_encoding)

        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = predictions

        return start_logits, end_logits, type_logits, sp_logits, start_position, end_position
