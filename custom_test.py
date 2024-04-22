import os
import soundfile as sf

from custom import inference, train

if __name__ == "__main__":
    # Test inference
    # synthesis_result = inference(
    #     "./test/SoVITS_weights/hello_e8_s160.pth",
    #     "./test/GPT_weights/hello-e15.ckpt",
    #     "./test/ref/segment_1.wav",
    #     "微笑是这个世界上最容易的事情。",
    #     "zh",
    #     "先帝创业未半，而中道崩殂",
    #     "zh",
    # )
    #
    # with open("output.wav", "wb") as f:
    #     generator = synthesis_result
    #     for audio_data in generator:
    #         f.write(audio_data)

    # Test train
    train(
        "name",
        "opt",
        "./test/output/asr/slices.list",
        "./test/output/slices",
        6, 8, 16, 8,
    )
