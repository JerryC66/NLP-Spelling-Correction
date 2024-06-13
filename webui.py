# 命令行 streamlit run webui.py运行

import streamlit as st
from annotated_text import annotated_text
import collections
import nltk
from nltk.corpus import reuters
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
num = 0
# 要确保先下载好
nltk.download('reuters')
nltk.download('punkt')


# 加载词汇表
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        vocab = set(file.read().split())
    return vocab


# 数据预处理
def load_data(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentence_id, error_count, sentence = line.strip().split('\t')
            sentences.append((sentence_id, int(error_count), sentence))
    return sentences


# 检查候选单词是否在词汇表中
def check(candidate_list, vocab):
    return [word for word in candidate_list if word in vocab]


# 信道模型：常见编辑操作
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return_set = set(deletes + transposes + replaces + inserts)
    for letter in word:
        if letter.isupper() and letter == word[0].upper():
            return [item[0].upper() + item[1:] for item in return_set]
    return return_set


def edits1_with_upper_letter(word):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return_set = set(deletes + transposes + replaces + inserts)
    return return_set


def build_language_model():
    words = reuters.words()
    trigrams = ngrams(words, 3)
    bigrams = ngrams(words, 2)
    unigrams = words

    trigram_freq = collections.Counter(trigrams)
    bigram_freq = collections.Counter(bigrams)
    unigram_freq = collections.Counter(unigrams)

    return trigram_freq, bigram_freq, unigram_freq


def trigram_probability(trigram_freq, bigram_freq, w1, w2, w3):
    w1_w2_freq = bigram_freq[(w1, w2)]
    return trigram_freq[(w1, w2, w3)] / w1_w2_freq if w1_w2_freq > 0 else 0


def bigram_probability(bigram_freq, w1, w2):
    w1_freq = sum(freq for (first_word, _), freq in bigram_freq.items() if first_word == w1)
    return bigram_freq[(w1, w2)] / w1_freq if w1_freq > 0 else 0


def unigram_probability(unigram_freq, word):
    total_words = sum(unigram_freq.values())
    return unigram_freq[word] / total_words if total_words > 0 else 0


def generate_upper_candidates(word, vocab):
    upper_candidate_list = edits1_with_upper_letter(word)
    checked_upper_candidates = check(upper_candidate_list, vocab)
    if checked_upper_candidates:
        return checked_upper_candidates
    else:
        further_upper_candidates = []
        for candidate in upper_candidate_list:
            further_edits = edits1_with_upper_letter(candidate)
            further_upper_candidates.extend(further_edits)
        checked_further_upper_candidates = check(further_upper_candidates, vocab)
        if checked_further_upper_candidates:
            return checked_further_upper_candidates
        # 如果两轮都没有找到，返回空列表
        print('cannot edit: ' + word)
        return []


def generate_candidates(word, vocab):
    # 首先生成第一轮编辑后的候选词
    candidate_list = edits1(word)

    # 检查这些候选词是否在词汇表中
    checked_candidates = check(candidate_list, vocab)

    # 如果在词汇表中找到了候选词，直接返回这些候选词
    if checked_candidates:
        return checked_candidates

    # 如果第一轮没有找到，进行第二轮编辑操作
    else:
        # 对第一轮的每个候选词进行进一步编辑
        further_candidates = []
        for candidate in candidate_list:
            further_edits = edits1(candidate)
            further_candidates.extend(further_edits)

        # 检查第二轮编辑后的候选词是否在词汇表中
        checked_further_candidates = check(further_candidates, vocab)

        # 如果在词汇表中找到了候选词，返回这些候选词
        if checked_further_candidates:
            return checked_further_candidates
        else:
            generate_upper_candidates(word, vocab)


def check_if_skip(word, check_set):
    if word in check_set or not word.isalpha() or word == "'s":
        return True
    else:
        return False


# 这里假设 edits1 和 check 函数已经被定义
# edits1 函数用于生成给定单词的编辑候选词
# check 函数用于检查给定的候选词列表中哪些词在词汇表中


def correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab):
    words = nltk.word_tokenize(sentence)  # 使用NLTK的word_tokenize分词，可以处理标点符号
    corrected_sentence = []
    non_word_errors = set()
    count = 0

    # First pass: correct non-word errors
    for i, word in enumerate(words):
        if check_if_skip(word, vocab):  # 保留标点符号或非字母字符
            corrected_sentence.append(word)
        else:
            count += 1
            non_word_errors.add(word)
            candidate_list = generate_candidates(word, vocab)

            if candidate_list:
                if i > 1:
                    previous_two_words = corrected_sentence[-2:]
                    best_candidate = max(candidate_list, key=lambda w: trigram_probability(trigram_freq, bigram_freq,
                                                                                           previous_two_words[0],
                                                                                           previous_two_words[1], w))
                elif i == 1:
                    previous_word = corrected_sentence[-1]
                    best_candidate = max(candidate_list,
                                         key=lambda w: bigram_probability(bigram_freq, previous_word, w))
                else:
                    best_candidate = max(candidate_list, key=lambda w: unigram_probability(unigram_freq, w))
                corrected_sentence.append(best_candidate)
            else:
                corrected_sentence.append(word)

    real_word_count = error_count - count
    # print(corrected_sentence + [real_word_count])
    if real_word_count > 0:
        prob = []
        for i, word in enumerate(corrected_sentence):
            if check_if_skip(word, non_word_errors):
                continue  # Skip words that were already corrected as non-word errors
            probability = unigram_probability(unigram_freq, word)
            # if i == 0:
            #     probability = unigram_probability(unigram_freq, word)
            # elif i == 1:
            #     previous_word = corrected_sentence[0]
            #     probability = bigram_probability(bigram_freq, previous_word, word)
            # else:
            #     previous_two_words = corrected_sentence[i - 2:i]
            #     probability = trigram_probability(trigram_freq, bigram_freq, previous_two_words[0],
            #                                       previous_two_words[1], word)
            prob.append((probability, i, word))

        prob.sort()
        # print(prob)
        # Correct the real-word errors with the lowest probabilities
        for _, i, word in prob:
            if real_word_count == 0:
                break
            candidate_list = generate_candidates(word, vocab)
            if candidate_list:
                if i > 1:
                    previous_two_words = corrected_sentence[i - 2:i]
                    best_candidate = max(candidate_list, key=lambda w: trigram_probability(trigram_freq, bigram_freq,
                                                                                           previous_two_words[0],
                                                                                           previous_two_words[1], w))
                elif i == 1:
                    previous_word = corrected_sentence[0]
                    best_candidate = max(candidate_list,
                                         key=lambda w: bigram_probability(bigram_freq, previous_word, w))
                else:
                    best_candidate = max(candidate_list, key=lambda w: unigram_probability(unigram_freq, w))
                if corrected_sentence[i] == best_candidate:
                    continue
                else:
                    corrected_sentence[i] = best_candidate
                    real_word_count = real_word_count - 1

    return ' '.join(corrected_sentence)


# 需要加一下缓存 不然每次交互重新加载很耗时
@st.cache_data
def load_model_and_data():
    vocab_path = 'vocab.txt'
    vocab = load_vocab(vocab_path)
    trigram_freq, bigram_freq, unigram_freq = build_language_model()
    return vocab, trigram_freq, bigram_freq, unigram_freq


def main():
    st.subheader("Spelling Checker", divider='rainbow')

    vocab, trigram_freq, bigram_freq, unigram_freq = load_model_and_data()

    text_input = st.text_area('Input your text', height=100)
    error_count = st.number_input('Error count', min_value=0, max_value=10, value=0)

    if st.button("Correct"):
        st.divider()
        if text_input and error_count >= 0:
            corrected_text = correct_sentence(text_input, error_count, trigram_freq, bigram_freq, unigram_freq, vocab)
            st.write("Correct result:")
            show_differences(text_input, corrected_text)
        else:
            st.error("Please input valid information")


# 标注修改过的单词 用的Annotated Text Component for Streamlit
def show_differences(original_text, corrected_text):
    original_words = word_tokenize(original_text)
    corrected_words = word_tokenize(corrected_text)

    annotated_result = []
    for original, corrected in zip(original_words, corrected_words):
        if original.lower() != corrected.lower():
            annotated_result.append((corrected, "", "#faa"))
        else:
            annotated_result.append(corrected)
        annotated_result.append(" ")

    annotated_text(*annotated_result)


if __name__ == "__main__":
    main()
