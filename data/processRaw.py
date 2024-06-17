import os
import pickle as pkl

from sklearn.model_selection import train_test_split


def formatData(src_dir: str, dst_file: str) -> None:
    ls_filename = os.listdir(src_dir)
    data, label = [], []
    bag_of_word = set()
    bag_of_tag = set()

    IN_SENTENCE = False
    sentence, tag_seq = [], []

    for filename in ls_filename:
        with open(src_dir + filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    # 写入新句子数据
                    if IN_SENTENCE:
                        data.append(sentence)
                        label.append(tag_seq)
                        sentence, tag_seq = [], []
                        IN_SENTENCE = False
                    continue

                if not IN_SENTENCE:
                    IN_SENTENCE = True
                try:
                    word, tag = line.split()
                except Exception as e:
                    continue
                bag_of_word.add(word)
                bag_of_tag.add(tag)
                sentence.append(word)
                tag_seq.append(tag)

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.3, random_state=1, shuffle=True)
    with open(dst_file, 'wb') as f:
        pkl.dump((train_data, test_data, train_label, test_label, bag_of_word, bag_of_tag), f)


if '__main__' == __name__:
    src = './_extracted_brown/'
    dst = './data.pkl'
    formatData(src, dst)