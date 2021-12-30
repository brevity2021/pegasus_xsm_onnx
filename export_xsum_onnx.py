import os
import sys
import numpy as np
import time

import torch
from transformers import PegasusTokenizer
from transformers import PegasusForConditionalGeneration
import torch.nn.functional as F

import onnx
import onnxruntime as ort

model_name="google/pegasus-xsum"
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
encoder = pegasus_model.model.encoder
decoder = pegasus_model.model.decoder
lm_head = pegasus_model.lm_head

tokenizer = PegasusTokenizer.from_pretrained(model_name)
dummy_input = tokenizer("This is an amazing sentence.", return_tensors='pt')

def export_encoder(model, inp, exported_model_path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model,
                           inp,
                           exported_model_path,
                           export_params=True,
                           opset_version=14,
                           input_names=['input_ids'],
                           output_names=['hidden_states'],
                           dynamic_axes={
                               'input_ids': {0:'batch', 1: 'sequence'},
                               'hidden_states': {0:'batch', 1: 'sequence'},
                           })
        

class DecoderWithLMHead(torch.nn.Module):    
    def __init__(self, decoder, lm_head, final_logits_bias):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias
        
    def forward(self, input_ids, encoder_hidden_states):
        outputs = self.decoder(input_ids=input_ids,
                               attention_mask=None,
                               encoder_hidden_states=encoder_hidden_states)
        logits = self.lm_head(outputs[0]) + self.final_logits_bias
        next_token_logits = logits[:, -1, :]
        log_softmax = F.log_softmax(next_token_logits, 1)
        topk = torch.topk(log_softmax, 5, largest=True)
        return  topk.values, topk.indices
      
 def export_decoder(model, model_inputs, encoded, exported_model_path):
    model.eval()
    with torch.no_grad():
        _ = torch.onnx.export(model,
                          (model_inputs, encoded),
                          exported_model_path,
                          export_params=True,
                          opset_version=14,
                          input_names=['input_ids', 'encoder_hidden_states'],
                          output_names=['log_softmax', 'indices'],
                          dynamic_axes={
                              'input_ids': {0:'batch', 1: 'sequence'},
                              'encoder_hidden_states': {0:'batch', 1: 'sequence'},
                              'log_softmax': {0:'batch'},
                              'indices': {0:'batch'},
                          })
 
export_encoder(encoder, dummy_input['input_ids'], output_encoder_path)
decoder_lm_head = DecoderWithLMHead(decoder, lm_head, pegasus_model.final_logits_bias)
last_state = encoder(input_ids=dummy_input['input_ids']).last_hidden_state
decoder_inputs = torch.tensor([[0]])
export_decoder(decoder_lm_head, decoder_inputs, last_state, output_decoder_path)
