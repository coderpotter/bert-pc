from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from tqdm import tqdm
import pandas as pd
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# # testing the tokenizer
# print(tokenizer.convert_ids_to_tokens(ids=tokenizer.encode('(AND (OR -s15 s10 s0) (OR -s19 -s4 -s16 -s15 -s12 -s10 s3 s2 -s18 -s20 s0) (OR s6 -s7 s13 s5 s0) (OR s16 s2 s10 -s4 -s14 -s9 -s11 -s19 -s6 s0) (OR s8 -s11 s0) (OR s14 s3 -s16 -s10 s2 -s18 s0) (OR -s19 s13 -s5 -s8 s0) (OR -s6 s5 -s19 -s7 s15 s0) (OR s20 s12 -s16 -s19 s9 s0) (OR -s11 -s13 s12 s7 s0) (OR -s20 s8 s17 s0) (OR -s1 -s11 s7 s0) (OR -s9 s11 s17 -s5 s0) (OR s16 s5 -s20 s0) (OR -s12 -s16 s13 s0) (OR -s1 -s9 s10 s0) (OR -s20 -s16 s3 -s5 s0) (OR s6 s11 s2 -s14 -s1 -s8 -s4 s0) (OR -s16 s7 -s20 s0) (OR -s11 s10 -s16 -s19 s0) (OR s4 s20 s7 s0) (OR s15 -s11 s0) (OR -s11 -s17 s0) (OR s10 -s8 -s6 s5 s0) (OR -s13 s9 s16 -s3 s0) (OR -s20 s18 -s10 -s7 s0) (OR s5 -s2 s0) (OR s20 s11 -s6 -s15 s0) (OR s13 -s19 s17 s0) (OR -s18 -s9 s11 -s6 s0) (OR s14 -s7 s0) (OR -s16 s2 s0) (OR -s9 -s11 -s12 -s8 s0) (OR -s1 -s4 s0) (OR -s17 s10 s13 s0) (OR -s11 s13 -s16 -s9 s0) (OR -s17 -s12 s10 s9 -s20 s16 s0) (OR -s10 s4 s12 s0) (OR -s7 -s3 s0) (OR s13 -s20 s0) (OR -s14 -s1 -s20 -s11 -s7 s18 s0) (OR -s20 s14 s6 -s3 -s18 -s7 -s1 s0) (OR s4 s9 -s7 s20 -s3 s5 s0) (OR -s5 -s14 -s9 s16 -s20 s0) (OR -s9 s16 -s11 s0) (OR -s13 -s11 s3 -s10 s0) (OR -s14 s7 s10 -s19 s0) (OR s7 s5 -s18 -s19 s0) (OR -s6 s16 -s5 s0) (OR -s8 -s20 s17 -s9 -s11 -s1 s0) (OR s12 s7 s0) (OR s3 -s20 s10 s0) (OR s8 -s4 s15 -s11 s0) (OR s14 -s9 s10 -s20 s6 s0) (OR -s16 s7 s10 s0) (OR s9 -s10 -s14 s0) (OR -s15 s11 -s17 -s4 s20 s9 -s10 -s6 s0) (OR -s8 s15 s1 s0) (OR -s2 s1 -s13 -s11 -s17 s0) (OR -s4 -s11 s6 -s17 s0) (OR -s8 s7 -s14 s10 -s9 s0) (OR s8 -s16 s6 -s10 s14 s0) (OR s16 -s2 -s3 s0) (OR s7 s12 -s16 s0) (OR -s18 -s13 s14 s6 s0) (OR s18 -s1 -s12 -s8 s0) (OR -s20 s12 s0) (OR s5 -s17 s16 s0) (OR -s9 s18 s11 s17 s0) (OR s10 -s14 s6 s11 s0) (OR s6 -s17 s7 s1 -s5 -s11 s0) (OR -s5 -s16 -s11 s0) (OR s12 -s17 -s2 s0) (OR s13 s19 -s14 s8 -s17 s6 s0) (OR -s19 s2 -s7 s0) (OR s16 s9 s14 s0) (OR s15 s11 -s2 s20 -s12 -s9 -s13 s1 s0) (OR s11 s2 s0) (OR s3 -s15 s20 s6 s0) (OR -s17 s1 -s15 s10 s0) (OR s13 s5 -s12 s0) (OR s13 s16 -s1 s11 -s6 s0) (OR s10 -s4 -s12 -s19 -s20 s0) (OR -s4 s2 s11 s0) (OR s9 -s20 -s5 s0) (OR -s18 s9 s6 s11 -s1 s12 s0) (OR s1 s18 -s11 -s7 -s20 s15 s0) (OR -s1 s6 -s17 -s16 s0) (OR -s14 -s1 s20 -s17 -s12 s0) (OR -s10 -s15 -s3 s19 s0) (OR s5 -s15 -s17 s0) (OR s16 s5 s3 s0) (OR s10 -s13 -s18 -s6 -s11 -s3 s0) (OR -s20 -s9 s13 s0) (OR -s6 -s1 s0) (OR -s11 s5 s19 s0) (OR -s13 s4 s7 -s6 s0) (OR s11 -s20 -s2 -s10 s5 -s13 -s6 s0) (OR s12 -s20 s0) (OR -s20 -s15 s4 -s17 -s16 s9 -s18 -s6 -s5 s0) (OR -s8 -s9 s16 s0) (OR -s13 s10 -s9 -s17 s12 s0) (OR -s8 -s19 -s3 s1 -s5 -s20 s0) (OR -s1 -s17 -s19 s15 s0) (OR s18 -s11 s8 -s3 -s5 s0) (OR s16 -s7 s0) (OR -s10 s20 s15 -s3 s6 s5 s0) (OR s7 s2 s16 -s3 -s4 s18 s1 s0) (OR s18 s19 s10 -s8 s7 s20 -s13 s12 s16 s15 -s5 s11 s17 s0) (OR -s4 -s17 -s19 -s7 s0) (OR -s16 -s6 -s2 -s9 -s1 s0) (OR -s20 s13 s9 s0) (OR -s19 s14 s5 s18 -s16 s0) (OR s7 s10 s16 s0) (OR s5 s20 s2 s0) (OR s3 -s15 s1 -s14 s0) (OR -s16 s12 s1 s0) (OR -s17 s20 s0) (OR -s14 s9 s17 s15 s0) (OR s3 s4 -s14 s19 s0) (OR s6 -s2 -s15 -s9 s0) (OR -s11 s10 s0) (OR -s5 -s1 s17 -s10 s16 s0) (OR s10 -s8 -s14 s6 -s4 s0) (OR -s20 s18 s14 -s5 s0) (OR s17 s11 -s18 -s10 s9 -s16 s0) (OR s5 s16 s11 s12 s0) (OR s4 -s18 s2 s12 s0) (OR -s2 s3 -s4 s0) (OR s2 s8 s15 s0) (OR s15 s4 s9 s0) (OR s15 -s14 s18 s19 -s11 s0) (OR -s16 s8 s0) (OR s14 -s17 -s7 s13 -s8 -s12 s0) (OR s14 s16 s17 -s11 -s2 s0) (OR -s18 s9 s17 -s1 -s19 -s20 s12 -s14 -s15 s0) (OR -s12 s6 s4 -s2 s20 -s16 s14 s0) (OR -s8 -s17 s15 -s6 -s2 s0) (OR s1 -s10 -s5 s0))',
#                                                            add_special_tokens=False),
#                                       skip_special_tokens=True))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased').to('cuda:1')

class SATDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_df = pd.concat([pd.read_csv('train/grp1/train.csv', index_col=0), pd.read_csv('train/grp2/train.csv', index_col=0)]).rename(columns={'SAT':'labels'})
train_texts = train_df['S-exp'].to_list()
train_labels = train_df['labels'].astype(np.float16).to_list()
print('encoding train data')
train_encodings = tokenizer(train_texts, truncation=True, padding=True).to('cuda:1')
torch.save(train_encodings, 'train_encodings.pt')
train_dataset = SATDataset(train_encodings, train_labels)
torch.save(train_dataset, 'train.pt')

# eval_df = pd.concat([pd.read_csv('test/grp1/test.csv', index_col=0), pd.read_csv('test/grp2/test.csv', index_col=0)]).rename(columns={'SAT':'labels'})
# eval_texts = eval_df['S-exp'].to_list()
# eval_labels = eval_df['labels'].astype(np.float16).to_list()
# print('encoding eval data')
# eval_encodings = tokenizer(eval_texts, truncation=True, padding=True).to('cuda:1')
# torch.save(eval_encodings, 'eval_encodings.pt')
# eval_dataset = SATDataset(eval_encodings, eval_labels)
# torch.save(eval_dataset, 'eval.pt')