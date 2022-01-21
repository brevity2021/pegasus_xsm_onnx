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
export_text = "This is an amazing sentence."

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

def export_encoder_and_decoder(tokenizer, model, export_text, output_encoder_path, output_decoder_path):
    export_input = tokenizer(export_text, return_tensors='pt')
    export_encoder(model.model.encoder, export_input['input_ids'], output_encoder_path)
    decoder_lm_head = DecoderWithLMHead(model.model.decoder, model.lm_head, model.final_logits_bias)
    export_decoder(decoder_lm_head, export_input['input_ids'], model.model.encoder(input_ids=export_input['input_ids']
export_encoder_and_decoder(ori_tokenizer, ori_pegasus_model, export_text, output_encoder_path, output_decoder_path)
