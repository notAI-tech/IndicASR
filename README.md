# IndicASR
ASR for Indian Languages

# Use as python module
```bash
pip install --upgrade indicasr
```

```python
from indicasr import IndicASR
asr = IndicASR("telugu")
# Run one file at once
asr.transcribe("samples/telugu/hari.16k.wav")
# "ఈ సినిమా తర్వాత నిర్మాతలు రూటు మార్చే ఆలోచనలో ఉన్నారు"

# Batch inference
asr.transcribe(["samples/telugu/hari.16k.wav",
              "samples/telugu/ramana.16k.wav"])
# ["ఈ సినిమా తర్వాత నిర్మాతలు రూటు మార్చే ఆలోచనలో ఉన్నారు",
# "భారత దేశము నా మాత్ర భూమి భారతీవులంతా నా సోదరి సోదరులు"]
```

|sample name   |  prediction    |  expected    |
|--------|:--------------:|:--------------:
|telugu/hari.16k.wav | ఈ సినిమా తర్వాత నిర్మాతలు రూటు మార్చే ఆలోచనలో ఉన్నారు    | ఈ సినిమా తర్వాత నిర్మాతలు రూటు మార్చే ఆలోచనలో ఉన్నారు |
|telugu/harsha.16k.wav| నేను ఇప్పుడు గడ్డి కొడుతున్నారు    | నేను ఇప్పుడు గడ్డి కొడుతున్నాను |
|telugu/indra.16k.wav| నేను భారత దేశంలో ఉన్నాను    | నేను భారత దేశంలో ఉన్నాను |
|telugu/praneeth.16k.wav| నా పేరు ప్రణి బేదపూడి    | నా పేరు ప్రణీత్ బేదపూడి   |
|telugu/ramana.16k.wav| భారత దేశము నా మాత్ర భూమి భారతీవులంతా నా సోదరి సోదరులు    |  భారత దేశము నా మాతృ భూమి భారతీయులంతా నా సోదరీ సోదరులు|
|telugu/sai_krishna.16k.wav| నా పేరు సాయి కృష్ణ    | నా పేరు సాయి కృష్ణ    |
