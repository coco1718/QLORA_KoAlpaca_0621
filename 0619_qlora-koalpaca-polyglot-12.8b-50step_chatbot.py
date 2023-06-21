

#!nvidia-smi
#!pip install -q -U bitsandbytes pip install -q bitsandbytes
#!pip install -q -U git+https://github.com/huggingface/transformers.git
#!pip install -q -U git+https://github.com/huggingface/peft.git
#!pip install -q -U git+https://github.com/huggingface/accelerate.git
#!pip install -q datasets

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import PeftModel, PeftConfig

peft_model_id = "beomi/qlora-koalpaca-polyglot-12.8b-50step"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()





# https://www.youtube.com/watch?v=WTul6LIjIBA 참고
# pip install gTTS
# pip install playsound==1.2.2
#cmd명령어
#C:\AISpeaker>python -m venv myenv 가상환경만들기
#C:\AISpeaker>.\myenv\Scripts\activate 가상환경들어가기
#(myenv) C:\AISpeaker>pip install gTTS
#(myenv) C:\AISpeaker>pip install playsound==1.2.2py#마이크설치
#(myenv) C:\AISpeaker>pip install SpeechRecognition
#(myenv) C:\AISpeaker>pip install PyAudio



# 음성인식 (듣기)
def listen(recognizer, audio):
    try:
        sentence = recognizer.recognize_google(audio, language='ko')
        print('[나]' + sentence)
        answer(sentence)


    except sr.UnknownValueError:
        print('인식 실패')  # 음성인식실패
    except sr.RequestError as e:  # 네트워크오류
        print('요청실패 : {0}'.format(e))  # API Key오류, 네트워크단절 등

def answer(sentence):
    q = f"### 질문: {sentence}\n\n### 답변:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=200,
        #max_new_tokens=50,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )

    speak(tokenizer.decode(gened[0]))


# 소리내어읽기
def speak(text):

    print('[인공지능]' + text)
    file_name = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(file_name)
    playsound(file_name)
    if os.path.exists(file_name):  # voice.mp3 파일삭제
        os.remove(file_name)


# def 함수끝에 결과값을 return하기도 하고, speak(return값) 함수를 부르기도 함

r = sr.Recognizer()
m = sr.Microphone()
speak('무엇을 도와줄까?')


# 백그라운드에서 듣고 있음 https://smart-factory-lee-joon-ho.tistory.com/357
stop_listening = r.listen_in_background(m, listen)
#stop_listening(wait_for_stop=False) # 더 이상 듣지 않음,듣기를듣지마란뜻

while True:  #무한루프
   time.sleep(0.1)

