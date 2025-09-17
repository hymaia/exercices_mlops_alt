import string
import sys
from collections import Counter


def main():
    input_str = sys.argv[1]
    cleaned_input = clean_text(input_str)
    print(wordcount(cleaned_input))


def clean_text(input_str: str) -> str:
    cleaned_str = input_str
    for character in string.punctuation:
        cleaned_str = cleaned_str.replace(character, " ")
    return cleaned_str.lower()


def wordcount(input_str: str):
    word_list = input_str.split()
    return Counter(word_list).most_common()


def word_count_simple(input_str: str):
    word_list = input_str.split()
    return len(word_list)


if __name__ == "__main__":
    main()