import torch
from transformers import pipeline
from auto_gptq import AutoGPTQForCausalLM

from utils.prompter import Prompter

MODEL = "j5ng/kullm-12.8b-GPTQ-8bit"
model = AutoGPTQForCausalLM.from_quantized(MODEL, device="cuda:1", use_triton=False)

pipe = pipeline('text-generation', model=model,tokenizer=MODEL)

        num_beams=5,
prompter = Prompter("kullm")

def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(
        prompt, max_length=512,
        temperature=0.2,
        repetition_penalty=3.0,
        eos_token_id=2
    )
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result

instruction = """
백종원은 1966년 충남 예산군에서 집안의 종손으로 태어났다. 중학교 시절 상경해 강남 8학군 서울고등학교[23]를 졸업하고 연세대학교에 입학했다. 고등학교 졸업 직후엔 잠시 서울특별시 장한평에 위치한 중고차 시장에서 자동차 중개업자로 활동하기도 했다.
2019년, KBS의 토크쇼 프로그램인 대화의 희열에 출연하여 어렸을 적 이야기를 많이 했는데, 만석꾼이었던 증조할아버지 백영기(白榮基)의 피를 이어받은 영향인지 어렸을 때부터 장사꾼 기질이 있다고 스스로 자각하고 있었다고 한다.
9살 때에는 산에 놀러갔다가 본 버섯 농장에서 별다른 투자도 안 한 거 같은데 돈이 된다는 이야기를 듣고 꿈을 버섯 농사로 정한 적도 있었고, 초등학교 4학년때는 캔이 아닌 병에 음료가 나올 시절에 음료수 병을 보고 '저게 돈이 될 것 같다'고 생각해 학교 리어카를 빌려 오락 시간과 보물찾기 같은 시간을 다 건너뛰고 리어카 6개 분량의 공병을 모아서 고물상에 갖다 팔아 큰 돈을 벌었다고 한다.
그리고 5학년 1학기까지 이렇게 돈을 벌었고, 방위성금으로 다 냈다고 한다. 
"""
result = infer(instruction=instruction, input_text="백종원의 증조할아버지는 누구?")
print(result)

instruction = """
이번 전세사기 특별법에는 조세 채권 안분을 비롯, 전세 사기 피해자에 우선매수권 부여, LH공사 공공임대 활용 등의 내용이 포함됐다.
전세사기로 피해를 입은 세입자로 분류되면 살고 있는 집이 경매로 넘어갔을 경우 우선매수권을 부여받고,
경매로 집을 낙찰받을 경우 금융지원을 받을 수 있다.
"""
result = infer(instruction=instruction, input_text="전세사기 특별법에는 어떤 내용이 포함되었나요?")
print(result)