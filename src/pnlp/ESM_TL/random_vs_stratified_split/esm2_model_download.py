from transformers import EsmTokenizer, EsmForMaskedLM

esm_version = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
model = EsmForMaskedLM.from_pretrained(esm_version, cache_dir="../../../.cache")
print(f"Model download ({esm_version}) successful!\n")

esm_version = "facebook/esm2_t30_150M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
model = EsmForMaskedLM.from_pretrained(esm_version, cache_dir="../../../.cache")
print(f"Model download ({esm_version}) successful!\n")

esm_version = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
model = EsmForMaskedLM.from_pretrained(esm_version, cache_dir="../../../.cache")
print(f"Model download ({esm_version}) successful!\n")