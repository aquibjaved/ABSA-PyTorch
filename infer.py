import numpy as np
import torch
import torch.nn.functional as F
from models.bert_spc import BERT_SPC
from transformers import BertModel
from data_utils import Tokenizer4Bert, pad_and_truncate
from dependency_graph import dependency_adj_matrix
import argparse


class BertAspectClassifier:
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location='cpu'))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]
        text_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = self.tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        text_len = np.sum(text_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence(
            '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        text_bert_indices = self.tokenizer.text_to_sequence(
            "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

        dependency_graph = dependency_adj_matrix(text)
        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary,
            'dependency_graph': dependency_graph,
        }

        # print(type(torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols))

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]

        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs, t_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_sent', help='provide text', required=True)
    parser.add_argument('--aspect', help='provide aspect tokens in list form [word1, word2,..]', nargs='+', required=True)
    parser.add_argument('--model_path', help='provide model path', required=True)

    args = parser.parse_args()

    targets = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    model_classes = {'bert_spc': BERT_SPC}
    input_colses = {

        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }


    class Option(object): pass


    opt = Option()
    opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = args.model_path #'/Users/aquibkhan/Desktop/bert_spc_restaurent_val_acc_0.7774'
    opt.max_seq_len = 85
    opt.bert_dim = 768
    opt.polarities_dim = 3
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.inputs_cols = input_colses[opt.model_name]

    inf = BertAspectClassifier(opt)
    text = args.text_sent
    aspcts = args.aspect

    for aspects in aspcts:
        t_probs, inputs_ = inf.evaluate(text.lower(), aspect=aspects)
        idx = t_probs.argmax(axis=-1) - 1
        print(aspects, targets[idx[0]])

