import pyopenjtalk
import regex
from urllib.request import urlretrieve
import os
import tarfile

user_dict = [
    ("友奈", "ゆうな"),
    ("犬吠埼風", "いぬぼうざきふう"),
    ("風先輩", "ふうせんぱい"),
    ("風殿", "ふうどの"),
    ("風様", "ふうさま"),
    ("風達", "ふうたち"),
    ("風以外", "ふう以外"),
    ("風さん", "ふうさん"),
    ("つむじ風", "つむじかぜ"),
    ("よい風", "よいかぜ"),
    ("すっごい風", "すっごいかぜ"),
    ("風を感じ", "かぜを感じ"),
    ("激しい風", "激しいかぜ"),
    ("激しい風", "激しいかぜ"),
    ("強い風", "強いかぜ"),
    ("果樹", "かじゅ"),
    ("神樹", "しんじゅ"),
    ("樹海", "じゅかい"),
    ("樹木", "じゅもく"),
    ("樹", "いつき"),
    ("夏凜", "かりん"),
    ("芽吹き", "めぶき"),
    ("芽吹く", "めぶく"),
    ("芽吹い", "めぶい"),
    ("芽吹", "めぶき"),
    ("伊予島", "いよじま"),
    ("杏", "あんず"),
    ("夕海子", "ゆみこ"),
    ("上里", "うえさと"),
    ("美森", "みもり"),
    ("秋原", "あきはら"),
    ("雪花", "せっか"),
    ("古波蔵", "こはぐら"),
    ("棗", "なつめ"),
    ("須美", "すみ"),
    ("水都", "みと"),
    ("真鈴", "ますず"),
    ("美佳", "よしか"),
    ("法花堂", "ほっけどう"),
    ("天の神", "てんのかみ"),
    ("象頭", "ぞーず"),
    ("五岳", "ごがく"),
    ("～", "ー"),
    ("〜", "ー"),
    ("...", "…"),
    ("..", "…"),
    (".", "…"),
    ("、", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
]

regex_dict = [
    (regex.compile(r"風([ぁ-ゖ])吹"), "かぜ{1}吹"),
    # [漢字]風 は漢字のまま残す。
    # 風[漢字] は漢字のまま残す。
    # それ以外は「ふう」にする。
    (regex.compile(r"(^|[^\p{Script=Han}])風([^\p{Script=Han}]|$)"), "{1}ふう{2}"),
    # 不要な記号
    (regex.compile(r"[「」『』（）{}]"), ""),
    # @s(60)みたいな制御文字
    (regex.compile(r"@[a-z]\(.*?\)"), ""),
]


def init():
    # 辞書をダウンロードしてくる
    # pyopenjtalk デフォルトの機能に任せると、Read only な場所にダウンロードしようとしてしまうため
    # https://github.com/r9y9/pyopenjtalk/blob/master/pyopenjtalk/__init__.py

    dict_dir = os.environ.get("OPEN_JTALK_DICT_DIR")
    dict_url = "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz"
    download_path = "/tmp/dic.tar.gz"
    extract_path = os.path.abspath(os.path.join(dict_dir, "../"))
    
    print('Downloading {} to {}'.format(dict_url, download_path))
    urlretrieve(dict_url, download_path)

    print("Extracting {} to {}".format(download_path, extract_path))
    with tarfile.open(download_path, mode="r|gz") as tar:
        tar.extractall(path=extract_path)
    
    os.remove(download_path)


def preprocess(text):
    for kanji, kana in user_dict:
        text = text.replace(kanji, kana)
    for before, after in regex_dict:
        text = regex.subf(before, after, text)
    return text


def encode(text):
    text = preprocess(text)
    phones = ""
    while 0 < len(text):
        symbol = ""
        match = regex.search(r"[,.!?…♪]", text)
        if match:
            length = match.span()[0]
            sub_text = text[:length]
            symbol = text[length]
            if 0 < len(sub_text):
                phones += pyopenjtalk.g2p(sub_text, kana=False) + " "
            symbol = text[length]
            phones += symbol + " "
            text = text[length+1:]
        else:
            length = len(text)
            phones += pyopenjtalk.g2p(text, kana=False)
            text = ""
    phones = phones.strip().replace(" ", "-").replace("--", "-")
    if len(phones.strip(",.!?…♪- ")) == 0:
        return None
    return phones
