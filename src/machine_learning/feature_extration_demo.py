from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def dict_demo():
    """
    字典特征提取
    :return:
    """
    data = [{'city': '北京'}, {'temperature': 100},
            {'city': '上海'}, {'temperature': 60},
            {'city': '深圳'}, {'temperature': 30}]

    # 字典特征提取
    # 1. 实例化
    transfer = DictVectorizer(sparse=False)

    # 2. 调用fit_transform
    trans_data = transfer.fit_transform(data)

    print(trans_data)


def english_count_text_demo():
    """
    文本特征提取--英文
    :return: None
    """
    data = ["life is short, you need python.",
            "life is too long, you dislike python."]

    # 1.实例化
    transfer = CountVectorizer(stop_words=['dislike'])
    # 2.调用fit_transform
    transfer_data = transfer.fit_transform(data)
    print(transfer.get_feature_names_out())
    print(transfer_data.toarray())
    print(transfer_data)


def chinese_count_text_demo():
    """
        文本特征提取--英文
        :return: None
        """
    data = ["人生 苦短,我 喜欢 python.",
            "生活 太长久,我 不 喜欢 python."]

    # 1.实例化
    transfer = CountVectorizer()
    # 2.调用fit_transform
    transfer_data = transfer.fit_transform(data)
    print(transfer.get_feature_names_out())
    print(transfer_data.toarray())


if __name__ == '__main__':
    # dict_demo()
    # english_count_text_demo()
    chinese_count_text_demo()
