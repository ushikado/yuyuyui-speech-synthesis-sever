import os
import json
import math
import datetime
import numpy as np
import soundfile
import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import phoneme_encoder

import flask
from google.cloud import storage, firestore

initialized = False

access_control_allow_origin = "https://ushikado.dithub.io"

document = firestore.Client().collection("speech-synthesis").document("speech-synthesis")

all_characters = [
    "結城 友奈", "東郷 美森", "犬吠埼 風", "犬吠埼 樹", "三好 夏凜",
    "乃木 園子", "鷲尾 須美", "三ノ輪 銀", "乃木 若葉", "上里 ひなた",
    "土居 球子", "伊予島 杏", "郡 千景", "高嶋 友奈", "白鳥 歌野",
    "藤森 水都", "秋原 雪花", "古波蔵 棗", "楠 芽吹", "加賀城 雀",
    "弥勒 夕海子", "山伏 しずく", "山伏 シズク", "国土 亜耶", "赤嶺 友奈",
    "弥勒 蓮華", "桐生 静", "安芸 真鈴", "花本 美佳",
]


def main(request):
    update_timestamp()
    update_cors_header()
    init()

    if request.method == "GET":
        return process_preflight(request)
    if request.method == "OPTIONS":
        return process_preflight(request)
    if request.method == "POST":
        return process_post(request)
        
    return process_preflight(request)


def update_timestamp():
    global document

    if "DEBUG" in os.environ:
        return

    document_dict = document.get().to_dict()
    document_dict["timestamp"] = datetime.datetime.now()
    document.set(document_dict)
    return


def update_cors_header():
    global document, access_control_allow_origin
    access_control_allow_origin = document.get().to_dict()["Access-Control-Allow-Origin"]
    return


def init():
    global initialized, hps, net_g, _
    if initialized:
        return

    phoneme_encoder.init()

    if not os.path.isfile("/tmp/model.pth"):
        client = storage.Client()
        client.download_blob_to_file("gs://speech-synthesis/yuyuyui-vits-model.pth", open("/tmp/model.pth", "wb"))
    
    hps = utils.get_hparams_from_file("configs/yuyuyui.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint("/tmp/model.pth", net_g, None)

    initialized = True


def process_preflight(request):
    global access_control_allow_origin
    headers = {
        'Access-Control-Allow-Origin': access_control_allow_origin
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    return ('', 204, headers)


def process_post(request):
    global access_control_allow_origin, hps, net_g, _
    
    request_json = request.get_json()
    chara = request_json["character_name"]
    text = request_json["text"]

    chara_id = all_characters.index(chara)
    sid = torch.LongTensor([chara_id])
    stn_tst = get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()

    tmp_file = "/tmp/voice.ogg"
    soundfile.write(tmp_file, audio, hps.data.sampling_rate, subtype="VORBIS", format="OGG")
    response = flask.make_response(flask.send_file(tmp_file, mimetype="audio/ogg", as_attachment=False))
    response.headers["Access-Control-Allow-Origin"] = access_control_allow_origin
    return response


def get_text(text, hps):
    phoneme_sequence = phoneme_encoder.encode(text, reject_nonverbal=False)
    text_norm = text_to_sequence(phoneme_sequence, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


if __name__ == "__main__":

    class RequestStub():
        def __init__(self, data):
            self.method = "POST"
            self.data = data
            return
        def get_json(self):
            print("RequestStub: get_json:", self.data)
            return self.data
    
    import sys
    print(main(RequestStub( {"character_name": sys.argv[1], "text": sys.argv[2]} )))
