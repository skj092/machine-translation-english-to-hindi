from torch.utils.data import Dataset
import torch


class BilingualDataset(Dataset):
    def __init__(self, ds, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, seq_len):
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len
        self.sos_token = torch.tensor(
            tgt_tokenizer.token_to_id("[SOS]"), dtype=torch.int64).unsqueeze(0)
        self.eos_token = torch.tensor(
            tgt_tokenizer.token_to_id("[EOS]"), dtype=torch.int64).unsqueeze(0)
        self.pad_token = torch.tensor(
            tgt_tokenizer.token_to_id("[PAD]"), dtype=torch.int64).unsqueeze(0)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]['translation']
        src_txt = row[self.src_lang]  # "this is a cat"
        tgt_txt = row[self.tgt_lang]

        src_index = self.src_tokenizer.encode(src_txt).ids
        tgt_index = self.tgt_tokenizer.encode(tgt_txt).ids

        num_pad_tokens = self.seq_len - len(src_index) - 2
        tgt_pad_tokens = self.seq_len - len(tgt_index) - 1

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_index, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * num_pad_tokens)], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_index, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_pad_tokens)], dim=0)

        label = torch.cat([
            torch.tensor(tgt_index, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_pad_tokens)], dim=0)

        # import code; code.interact(local=locals())
        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert len(label) == self.seq_len

        return {
            "encoder_input":  encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask==0
