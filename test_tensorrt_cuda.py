import os
import sys

import transformers
import torch
from transformers import PegasusTokenizer
import numpy as np
import argparse
import time

import onnx
import onnxruntime as ort

def create_ort_session_options(enable_profiling):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 0
    sess_options.log_verbosity_level = 1
    if enable_profiling:
        sess_options.enable_profiling = True
    return sess_options

def get_arg_parser():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--encoder_path', type=str, help='onnx encoder to be tested')
    parser.add_argument('--decoder_path', type=str, help='onnx decoder to be tested')
    parser.add_argument('--use_tensorrt', type=bool, default=False,help='use tensorrt or cuda only')
    parser.add_argument('--pegasus_model_dir',type=str,default="", help="pegasus model directory")
    parser.add_argument('--enable_profiling',type=bool,default=True, help="Whether to enable profiling")
    return parser
 
# naive summarization id generation
def get_summarization_ids(encoder_session, decoder_session, inputs, init_decoder_inputs, max_length):
    decoder_outputs = init_decoder_inputs
    current_length = 1
    encoder_output = encoder_session.run(None, {'input_ids':inputs["input_ids"].cpu().numpy()})
   # print("encoder output")
   # print(encoder_output)
    while current_length < max_length:
    #    print("Generating %d id" % current_length)
    #    print(decoder_outputs)
        (top_softmax, top_indices) = decoder_session.run(None, {'input_ids': decoder_outputs, "encoder_hidden_states": encoder_output[0]})
        print(top_softmax)
        print(top_indices)
        next_tokens = np.asarray([top_indices[0][0]])
        decoder_outputs = np.concatenate([decoder_outputs, next_tokens[:, None]], axis=-1)
        # end of sequence
        if next_tokens[0] == 1:
            break
        current_length+=1
    return decoder_outputs

def main():
  parser = get_arg_parser()
  args = parser.parse_args()
  EP_list_tensorrt = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
  EP_list_cuda = ['CUDAExecutionProvider']
  EP_list = EP_list_tensorrt if args.use_tensorrt else EP_list_cuda
  sess_options = create_ort_session_options(args.enable_profiling)
  encoder_session = ort.InferenceSession(args.encoder_path, sess_options, providers=EP_list)
  decoder_session = ort.InferenceSession(args.decoder_path, sess_options, providers=EP_list)
  #print(decoder_session.get_providers())
  model_name=args.pegasus_model_dir
  tokenizer = PegasusTokenizer.from_pretrained(model_name)
  test_text_array=["I have already taken classes in NLP, ML (both intro and grad level), and algorithms. I was even the teaching assistant for algorithms. I even was able to combine all of these in a self-project where I built a neural machine translation model capable of going from Shakespearean to modern English, which was able to make it all the way to the top of Hacker News.",
            "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.",
            "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.",
            "It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).",
            "Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."]
  warmup_text = "This is an amazing building."
  warmup_input=tokenizer([warmup_text], return_tensors='pt')
  decoder_inputs = [[0]]
  start = time.perf_counter()
  summarization_id_onnx = get_summarization_ids(encoder_session, decoder_session, warmup_input, decoder_inputs, 5)
  end = time.perf_counter()
  print(f'Warmup takes time: {(end-start)*1000} ms')
  for i in range(len(test_text_array)):
        test_input=tokenizer(test_text_array[i], truncation=True,max_length=1024, return_tensors='pt')
        start = time.perf_counter()
        summarization_id_onnx = get_summarization_ids(encoder_session, decoder_session, test_input, decoder_inputs, 5)
        end = time.perf_counter()
        print(f'Regular run takes time: {(end-start)*1000} ms')
        result_text = tokenizer.batch_decode(summarization_id_onnx, skip_special_tokens=True)
        print(result_text)
   encoder_prof_file = encoder_session.end_profiling()
   decoder_prof_file = decoder_session.end_profiling()
   print(f'Encoder profiling file:{encoder_prof_file}')
   print(f'Decoder profiling file:{decoder_prof_file}')              

if __name__=="__main__":
   main()
