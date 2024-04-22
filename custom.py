from webui import open1Ba, open1abc, open1Bb

from api import change_sovits_weights, change_gpt_weights, get_tts_wav


def inference(
        sovits_path,
        gpt_path,

        ref_wav_path,
        ref_text,
        ref_lang,
        text,
        lang,
):
    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)

    generator = get_tts_wav(
        ref_wav_path,
        ref_text,
        ref_lang,
        text,
        lang,
    )

    return generator


def train(name, opt_dir, inp_list_text, inp_wav_dir, sovits_batch, sovits_epoch, gpt_epoch, save_every_epoch,
          if_dpo=True):
    batch_size = sovits_batch
    exp_name = name
    text_low_lr_rate = 0.4
    if_save_latest = True
    if_save_every_weights = True
    gpu_numbers1Ba = '0-0'
    pretrained_s2G = "GPT_SoVITS/pretrained_models/s2G488k.pth"
    pretrained_s2D = "GPT_SoVITS/pretrained_models/s2D488k.pth"

    bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ssl_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    # prepare
    print("prepare")
    gen = open1abc(
        inp_list_text, inp_wav_dir,
        exp_name, gpu_numbers1Ba,
        gpu_numbers1Ba, gpu_numbers1Ba,
        bert_pretrained_dir, ssl_pretrained_dir,
        pretrained_s2G,
    )
    for i in gen:
        print(i)

    # sovits train
    print("sovits train")
    gen = open1Ba(
        batch_size, sovits_epoch,
        exp_name, text_low_lr_rate,
        if_save_latest, if_save_every_weights,
        save_every_epoch, gpu_numbers1Ba,
        pretrained_s2G, pretrained_s2D,
        opt_dir + "/SoVITS_weights"
    )
    for i in gen:
        print(i)

    # gpt train
    print("gpt train")
    pretrained_s1 = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    gen = open1Bb(
        1, gpt_epoch,
        exp_name, if_dpo,
        if_save_latest, if_save_every_weights,
        save_every_epoch, gpu_numbers1Ba,
        pretrained_s1, opt_dir + "/GPT_weights"
    )
    for i in gen:
        print(i)
