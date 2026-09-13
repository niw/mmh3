"""Writes tokenizer test cases: texts and the ids Hugging Face tokenizers gives them with the MiniMax H3 tokenizer.

Development-only tool. It needs the Hugging Face tokenizers package, which ComfyUI installs, and the tokenizer.json
of MiniMaxAI/MiniMax-H3, for example:

    python3 tools/golden/tokenizer_cases.py --tokenizer /path/to/MiniMax-H3/tokenizer/tokenizer.json \\
        --out tests/fixtures/tokenizer_cases.json

The tokenizer gets the same extra special tokens as ComfyUI's MiniMax H3 tokenizer, and nothing is added around the
text, as H3 conditions on the raw prompt.
"""

import argparse
import json

from tokenizers import Tokenizer

EXTRA_TOKENS = ["<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>"]

CASES = [
    "A red panda sips tea on a sunny wooden porch while birds chirp in the garden.",
    "integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. [Shot 2] At 00:04.500, "
    "the camera cuts to a close-up.\noverall_soundscape: A low, resonant hum.\nnon_diegetic_music: N/A",
    "I'm here, it'S fine. We'LL see; they're done, he'd go, you've won, she's 'quoted' 'Re'",
    "Numbers 12345 3.14159 1,000,000 -42 ١٢٣ ²³ ½ Ⅻ",
    "a  b\t\tc\n\nd \n e   \r\n f\n",
    "   leading and trailing   ",
    "   ",
    "\n",
    "赤いレッサーパンダが、日当たりの良い縁側でお茶を飲んでいる。鳥のさえずりが聞こえる。",
    "一只小熊猫在阳光明媚的木制门廊上喝茶，花园里鸟儿在歌唱。",
    "레서판다가 햇살 좋은 나무 현관에서 차를 마신다.",
    "Decomposed: é Å Å 각 が q̣̇",
    "Привет мир ΑΒΓ שלום مرحبا नमस्ते ภาษาไทย",
    "Emoji 🎬🐼✨ ★☆ — – “quotes” «guillemets» ‹›",
    "<|im_start|>user\nhello<|im_end|><|endoftext|>",
    "<d>x</d><|caption_start|>a caption<|caption_end|> <|lyrics_start|>la la<|lyrics_end|><|cutoff|>",
    "fn main() { println!(\"hi {}\", 1 + 2); } // https://example.com/path?q=1&r=2",
    "!!!???... ---- ____ $$$ %%% ^^^",
    "Fullwidth ＡＢＣ１２３ and no-break　spaces",
    "CamelCaseWord snake_case_word kebab-case-word",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    tokenizer = Tokenizer.from_file(arguments.tokenizer)
    tokenizer.add_special_tokens(EXTRA_TOKENS)
    assert [tokenizer.token_to_id(token) for token in EXTRA_TOKENS] == list(range(151669, 151676))
    cases = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids} for text in CASES]
    with open(arguments.out, "w") as file:
        file.write("[\n" + ",\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n]\n")
    print(f"wrote {len(cases)} cases, {sum(len(case['ids']) for case in cases)} tokens")


if __name__ == "__main__":
    main()
