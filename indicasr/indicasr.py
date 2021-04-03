from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset
import torchaudio
import torch
import os
import pydload
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

MODEL_URLS = {
    "te": {
        "pytorch_model.bin": "https://github.com/notAI-tech/IndicASR/releases/download/telugu/pytorch_model.bin",
        "config.json": "https://github.com/notAI-tech/IndicASR/releases/download/telugu/config.json",
        "special_tokens_map.json": "https://github.com/notAI-tech/IndicASR/releases/download/telugu/special_tokens_map.json",
        "preprocessor_config.json": "https://github.com/notAI-tech/IndicASR/releases/download/telugu/preprocessor_config.json",
        "tokenizer_config.json": "https://github.com/notAI-tech/IndicASR/releases/download/telugu/tokenizer_config.json",
        "vocab.json": "https://github.com/notAI-tech/IndicASR/releases/download/telugu/vocab.json",
    },
}

LANGUAGE_ALISASES = {
    "telugu": "te",
    # "tamil": "ta",
    # "english": "en",
    # "hindi": "hi",
    # "kannada": "kn",
    # "malayalam": "ml",
    # "marathi": "mr",
    # "punjabi": "pa",
    # "gujarati": "gu",
    # "bengali": "bn",
}


class IndicASR:
    tokenizer = None
    model = None

    def __init__(self, model_name='te'):
        model_name = model_name.lower()
        for x, y in LANGUAGE_ALISASES.items():
            model_name = model_name.replace(x, y)

        if model_name not in MODEL_URLS and model_name not in LANGUAGE_ALISASES:
            if model_name in LANGUAGE_ALISASES:
                model_name = LANGUAGE_ALISASES[model_name]

            print(f"model_name should be one of {list(MODEL_URLS.keys())}")
            return None

        home = os.path.expanduser("~")
        lang_path = os.path.join(home, ".IndicASR_" + model_name)
        if not os.path.exists(lang_path):
            os.mkdir(lang_path)

        for file_name, url in MODEL_URLS[model_name].items():
            file_path = os.path.join(lang_path, file_name)
            if os.path.exists(file_path):
                continue
            print(f"Downloading {file_name}")
            pydload.dload(url=url, save_to_path=file_path, max_time=None)

        self.processor = Wav2Vec2Processor.from_pretrained(lang_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(lang_path)

        if torch.cuda.is_available():
            print(f"Using GPU")
            self.model = self.model.cuda()

    def transcribe(
        self, input_file_paths
    ):
        return_single = True
        if isinstance(input_file_paths, list):
            return_single = False
        else:
            input_file_paths = [input_file_paths]

        ds = Dataset.from_dict({'path': [f for f in input_file_paths], 'sentence': ['' for f in input_file_paths]})
        def map_to_array(batch):
            speech, frame_rate = torchaudio.load(batch["path"])
            resampler = torchaudio.transforms.Resample(orig_freq=frame_rate, new_freq=16000)
            batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
            batch["sampling_rate"] = 16000
            batch["sentence"] = batch["sentence"]
            return batch
        
        ds = ds.map(map_to_array)

        def map_to_pred(batch):
            features = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
            device = 'cpu'
            if torch.cuda.is_available(): device = 'gpu'
            input_values = features.input_values.to(device)
            attention_mask = features.attention_mask.to(device)
            with torch.no_grad():
                logits = self.model(input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)

            batch["predicted"] = self.processor.batch_decode(pred_ids)
            batch["target"] = batch["sentence"]
            return batch

        result = ds.map(map_to_pred, batched=True, batch_size=16, remove_columns=list(ds.features.keys()))

        outputs = result['predicted']     

        if return_single:
            outputs = outputs[0]

        return outputs

if __name__ == '__main__':
    indicasr = IndicASR()
    while True:
        print('Enter')
        print(indicasr.transcribe(input().split()))
