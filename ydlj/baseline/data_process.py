from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer




class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.sent_spans = sent_spans
        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type
        self.token_to_orig_map=token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position




def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def read_examples( full_file):

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)    


    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    for case in tqdm(full_data):   
        key = case['_id']
        qas_type = "" # case['type']
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])   
        sup_titles = set([sp[0] for sp in case['supporting_facts']]) 
        orig_answer_text = case['answer']

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text=='unknown'  or orig_answer_text=="" # judge_flag??
        FIND_FLAG = False

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        # for debug
        titles = set()
        para_data=case['context']
        for paragraph in para_data:  
            title = paragraph[0]
            sents = paragraph[1]   


            titles.add(title)  
            is_gold_para = 1 if title in sup_titles else 0  

            para_start_position = len(doc_tokens)  

            for local_sent_id, sent in enumerate(sents):  
                if local_sent_id >= 100:  
                    break

                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)   
                sent_names.append(local_sent_name)  
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)   
                sent_id += 1   
                sent=" ".join(sent)
                sent += " "

                sent_start_word_id = len(doc_tokens)           
                sent_start_char_id = len(char_to_word_offset)  

                for c in sent:  
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1  
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))  

                # Answer char position
                answer_offsets = []
                offset = -1

                tmp_answer=" ".join(orig_answer_text)
                while True:

                    offset = sent.find(tmp_answer, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)   
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    FIND_FLAG = True   
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset   
                        end_char_position = start_char_position + len(tmp_answer) - 1  
                       
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])



               
                if len(doc_tokens) > 382:   
                   
                    break
            para_end_position = len(doc_tokens) - 1
            
            para_start_end_position.append((para_start_position, para_end_position, title, is_gold_para))  

        if len(ans_end_position) > 1:
            cnt += 1    
        if key <10:
            print("qid {}".format(key))
            print("qas type {}".format(qas_type))
            print("doc tokens {}".format(doc_tokens))
            print("question {}".format(case['question']))
            print("sent num {}".format(sent_id+1))
            print("sup face id {}".format(sup_facts_sent_id))
            print("para_start_end_position {}".format(para_start_end_position))
            print("sent_start_end_position {}".format(sent_start_end_position))
            print("entity_start_end_position {}".format(entity_start_end_position))
            print("orig_answer_text {}".format(orig_answer_text))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))
       
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position, 
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,   
            end_position=ans_end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    # max_query_length = 50

    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):  
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif example.orig_answer_text == 'unknown':
            ans_type = 3
        else:
            ans_type = 0   # 统计answer type

        query_tokens = ["[CLS]"]
        for token in example.question_text.split(' '):
            query_tokens.extend(tokenizer.tokenize(token))
        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[:max_query_length - 1]
        query_tokens.append("[SEP]")

        # para_spans = []
        # entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(query_tokens)

        all_doc_tokens = ["[CLS]"]   
        for token in example.question_text.split(' '):
            all_doc_tokens.extend(tokenizer.tokenize(token))
        if len(all_doc_tokens) > max_query_length - 1:
            all_doc_tokens = all_doc_tokens[:max_query_length - 1]
        all_doc_tokens.append("[SEP]")

        for (i, token) in enumerate(example.doc_tokens):    
            orig_to_tok_index.append(len(all_doc_tokens))  
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)    
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)  



        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
           
            if orig_start_position is None:  
                return 0, 0

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:  
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1  
            
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position = [], []
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):  
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            ans_start_position.append(s_pos)  
            ans_end_position.append(e_pos)

        

        for sent_span in example.sent_start_end_position:   
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue  
            sent_start_position = orig_to_tok_index[sent_span[0]] 
            sent_end_position = orig_to_tok_back_index[sent_span[1]] 
            sentence_spans.append((sent_start_position, sent_end_position)) 

        

        
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        
        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        sentence_spans = get_valid_spans(sentence_spans, max_seq_length)
       

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1
        if example.qas_id <10:
            print("qid {}".format(example.qas_id))
            print("all_doc_tokens {}".format(all_doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("doc_segment_ids {}".format(doc_segment_ids))
            print("query_tokens {}".format(query_tokens))
            print("query_input_ids {}".format(query_input_ids))
            print("query_input_mask {}".format(query_input_mask))
            print("query_segment_ids {}".format(query_segment_ids))
            print("sentence_spans {}".format(sentence_spans))
            print("sup_fact_ids {}".format(sup_fact_ids))
            print("ans_type {}".format(ans_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans,
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def get_valid_spans(spans, limit):
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", type=str, required=True)   
    parser.add_argument('--tokenizer_path',type=str,required=True)


    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    examples = read_examples( full_file=args.full_data)
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)











