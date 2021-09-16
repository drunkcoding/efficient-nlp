from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
model_name = "bert-base-uncased"
max_length = 512

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers","footers", "quotes"))
target_names=dataset.target_names
news_text = dataset.data
labels = dataset.target
(train_texts,valid_texts,train_labels,valid_labels)=train_test_split(news_text, labels, test_size=0.3)

print(type(train_texts))