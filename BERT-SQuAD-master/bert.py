from __future__ import absolute_import, division, print_function

import collections
import logging
import math

import numpy as np
import torch
from pytorch_transformers import( WEIGHTS_NAME, XLNetConfig,
                                  XLNetForQuestionAnswering, XLNetTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import (get_answer, input_to_squad_example,
                   squad_examples_to_features, to_list)

RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


class QA:

    def __init__(self, model_path: str):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 1000
        self.model, self.tokenizer = self.load_model(model_path)
        #self.cls_index = cls_index
        #self.p_mask = p_mask
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: str, do_lower_case=False):
        config = XLNetConfig.from_pretrained(model_path + "/config.json")
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        #print("token",tokenizer)
        model = XLNetForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer

    def predict(self, passage: str, question: str):
        example = input_to_squad_example(passage, question)
        features = squad_examples_to_features(example, self.tokenizer, self.max_seq_length, self.doc_stride,
                                              self.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index,all_cls_index, all_p_mask)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          #"token_type_ids": batch[2]
                          }
                example_indices = batch[3]
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                outputs = self.model(**inputs)
                #print(outputs)
            #print("Enter the for loop")
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(outputs[0][i]),
                                           start_top_index=to_list(outputs[1][i]),
                                           end_top_log_probs=to_list(outputs[2][i]),
                                           end_top_index=to_list(outputs[3][i]),
                                           cls_logits=to_list(outputs[4][i]))
                #print(result)
                all_results.append(result)
        #config = XLNetConfig.from_pretrained("C:\\Users\\JaiGatiri\\Downloads\\BERT-SQuAD-master\\model-oldversion\\config.json")
        #start_n_top, end_n_top = config.start_n_top, config.end_n_top
        #print("start")
        answer = get_answer(example, features, all_results, self.n_best_size, self.max_answer_length,self.model.config.start_n_top,self.model.config.end_n_top,self.tokenizer)
        print(answer)
        return answer