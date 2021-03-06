import os
import io
import json
import math
import datetime
import numpy as np
import soundfile
import torch
import wave
import lameenc

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import phoneme_encoder

if not "DEBUG" in os.environ:
    import flask
    from google.cloud import storage, firestore
    document = firestore.Client().collection("speech-synthesis").document("speech-synthesis")

initialized = False

# config
access_control_allow_origin = "https://ushikado.dithub.io"
bitrate = 64
model_gs_uri = "gs://speech-synthesis/yuyuyui-vits-model.pth"

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
    update_config()
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

    if not "DEBUG" in os.environ:
        document_dict = document.get().to_dict()
        document_dict["timestamp"] = datetime.datetime.now()
        document.set(document_dict)


def update_config():
    global document, access_control_allow_origin, bitrate, model_gs_uri

    if not "DEBUG" in os.environ:
        config = document.get().to_dict()
        access_control_allow_origin = config["Access-Control-Allow-Origin"]
        bitrate = config["bitrate"]
        model_gs_uri = config["model"]


def init():
    global initialized, encoder
    if initialized:
        return

    if not "DEBUG" in os.environ:
        phoneme_encoder.init()

    load_model()

    initialized = True


def load_model():
    global model_gs_uri,  hps, net_g
    
    if not "DEBUG" in os.environ:
        client = storage.Client()
        client.download_blob_to_file(model_gs_uri, open("/tmp/model.pth", "wb"))
    
    hps = utils.get_hparams_from_file("configs/yuyuyui.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    net_g.eval()

    utils.load_checkpoint("/tmp/model.pth", net_g, None)


def process_preflight(request):
    global access_control_allow_origin
    headers = {
        'Access-Control-Allow-Origin': access_control_allow_origin,
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    return ('', 204, headers)


def process_post(request):
    request_json = request.get_json()
    if "command" in request_json:
        return process_command(request_json)
    elif "character_name" in request_json and "text" in request_json:
        return process_synthesis(request_json)
    else:
        return process_bad_request(request_json)


def process_command(request_json):
    global access_control_allow_origin
    headers = {
        'Access-Control-Allow-Origin': access_control_allow_origin,
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    if request_json["command"] == "load_model":
        load_model()
    else:
        return process_bad_request(request_json)
        
    return ("ok", 200, headers)


def process_bad_request(request_json):
    global access_control_allow_origin

    response = flask.make_response(flask.send_file("bad_request.mp3", mimetype="audio/mp3", as_attachment=False))
    response.headers["Access-Control-Allow-Origin"] = access_control_allow_origin
    return response
    return ("", 400, headers)


def process_synthesis(request_json):
    global access_control_allow_origin, hps, net_g
    chara = request_json["character_name"]
    text = request_json["text"]

    try:
        chara_id = all_characters.index(chara)
    except ValueError:
        return process_bad_request(request_json)

    sid = torch.LongTensor([chara_id])
    stn_tst = get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()

    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, audio, hps.data.sampling_rate, subtype="PCM_16", format="WAV")
    wav_buffer.seek(0)
    with wave.open(wav_buffer) as wa:
        pcm = wa.readframes(wa.getnframes())
    mp3 = encode(pcm, hps.data.sampling_rate)
    
    if "DEBUG" in os.environ:
        return mp3

    response = flask.make_response(flask.send_file(io.BytesIO(mp3), mimetype="audio/mp3", as_attachment=False))
    response.headers["Access-Control-Allow-Origin"] = access_control_allow_origin
    return response


def get_text(text, hps):
    phoneme_sequence = phoneme_encoder.encode(text, reject_nonverbal=False)
    text_norm = text_to_sequence(phoneme_sequence, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def encode(pcm, sampling_rate):
    global bitrate

    encoder = lameenc.Encoder()
    encoder.silence()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sampling_rate)
    encoder.set_channels(1)
    encoder.set_quality(3)  # 2-highest, 7-fastest
    return encoder.encode(pcm) + encoder.flush()


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
    out = main(RequestStub( {"character_name": sys.argv[1], "text": sys.argv[2]} ))
    with open("out1.mp3", "bw") as fp:
        fp.write(out)
