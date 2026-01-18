from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load model and tokenizer (slow tokenizer for stability)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

LANGS = {
    "en": "eng_Latn",
    "ne": "npi_Deva",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "de": "deu_Latn",
}


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer.src_lang = LANGS[src_lang]

    inputs = tokenizer(text, return_tensors="pt")

    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANGS[tgt_lang]],
        max_length=256,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    print("Meta NLLB Offline Translator")
    print("Supported:", ", ".join(LANGS.keys()))

    src = input("From language code: ").strip()
    tgt = input("To language code: ").strip()
    text = input("Text to translate: ").strip()

    if src not in LANGS or tgt not in LANGS:
        print("Unsupported language code.")
        return
    
    translated = translate(text, src, tgt)
    print("Translated text:", translated)

if __name__ == "__main__":
    main()