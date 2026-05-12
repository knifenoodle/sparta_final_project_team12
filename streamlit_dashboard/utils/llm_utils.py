# import google.generativeai as genai
# import streamlit as st

# def get_total_insight(data_summary, search_query):
#     """
#     내부 리뷰 데이터 요약본과 외부(인스타그램) 트렌드 검색 쿼리를 바탕으로
#     통합 마케팅 전략 인사이트를 생성하는 AI 엔진입니다.
#     """
    
#     # 1. API 키 설정 (secrets.toml 연동)
#     genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
#     # 2. 모델 및 페르소나(시스템 인스트럭션) 설정
#     model = genai.GenerativeModel(
#         model_name='gemini-1.5-flash-latest',
#         system_instruction="""
#         당신은 Lululemon, Nike, Fila와 같은 글로벌 애슬레저 브랜드의 시니어 마케팅 전략가입니다. 
#         내부 리뷰 데이터 분석 결과와 실시간 소셜 트렌드를 결합하여 브랜드 점유율 확대를 위한 날카로운 전략을 제안하는 것이 당신의 역할입니다.
#         분석 시 '사용자 중심의 리얼 보이스(TPO, 인증샷 구도, 스타일링 등)'를 가장 중요하게 고려하세요.
#         답변은 반드시 제공된 마크다운(Markdown) 형식을 엄격히 준수하여 가독성을 극대화한 비즈니스 보고서 스타일로 작성하세요.
#         """
#     )
    
#     # 3. 사용자 중심의 통합 분석 프롬프트
#     prompt = f"""
#     [1. 내부 데이터 기반 현황 분석]
#     아래는 117만 건의 리뷰 데이터 중 현재 선택된 조건의 데이터 요약입니다.
#     {data_summary}

#     [2. 외부 실시간 트렌드 분석 요청]
#     키워드: "{search_query}"
#     위 키워드와 관련된 최신 인스타그램 소비자 트렌드를 분석해 주세요.

#     [작성 가이드 및 출력 형식]
#     반드시 응답의 **첫 번째 줄**에는 현재 트렌드를 가장 잘 반영하는 인스타그램 해시태그 3개만 '#' 기호 없이 작성하세요. (예: 고프코어룩, 애슬레저코디, 블록코어)
#     **두 번째 줄부터** 아래 양식에 맞춰 마크다운 리포트를 작성하세요.

#     반드시 아래의 양식에 맞춰 작성해 주세요. (이모지를 적절히 활용하여 시각적으로 돋보이게 하세요)

#     ### 🎯 통합 전략 요약
#     > **한 줄 요약:** 내부 데이터와 외부 트렌드를 관통하는 핵심 인사이트를 한 줄로 정의

#     ### 🗣️ 소비자 리얼 보이스 분석
#     * **주요 TPO 및 해시태그:** (일반 사용자들이 이 브랜드를 소비할 때 주로 사용하는 상황과 연관 해시태그)
#     * **스타일링 및 인증샷 트렌드:** (최근 유행하는 구도, 핏, 고프코어 등 스타일링 특징)
#     * **데이터 vs 소셜 갭(Gap):** (내부 리뷰 데이터의 불만/만족 포인트와 소셜 미디어 상의 이미지 간의 차이점이나 시너지)

#     ### 💡 액션 플랜 (Action Plan)
#     | 전략 구분 | 즉각 실행 아이템 | 기대 효과 |
#     | :--- | :--- | :--- |
#     | 마케팅/콘텐츠 | (구체적인 인스타그램 캠페인 제안 등) | (기대되는 바이럴 효과 등) |
#     | 제품/UX 개선 | (리뷰 데이터 기반 개선점 반영 등) | (리뷰 평점 상승 등) |
    
#     ---
#     *주의: 제공된 데이터 요약이 분석하기에 부족할 경우, 억지로 수치를 만들지 말고 데이터 보완 필요성을 명시하세요.*
#     """
    
#     # 4. AI 답변 생성 및 예외 처리
#     try:
#         response = model.generate_content(prompt)
#         lines = response.text.strip().split('\n')
        
#         # 첫 번째 줄에서 해시태그 3개를 가져와서 쉼표 기준으로 쪼개기 (리스트로 변환)
#         raw_hashtags = lines[0].strip().replace("#", "").split(',')
#         dynamic_hashtags = [tag.strip() for tag in raw_hashtags] # 공백 제거 후 리스트로 저장
        
#         full_report = '\n'.join(lines[1:])
        
#         # 이제 문자열 1개가 아니라, 리스트 형태의 해시태그 모음과 리포트를 반환합니다.
#         return dynamic_hashtags, full_report 

#     except Exception as e:
#         # 에러 시 기본값도 리스트 형태로 맞춰줍니다.
#         return ["운동복코디", "애슬레저", "애슬레저코디"], f"🚨 인사이트 생성 중 오류가 발생했습니다: {str(e)}"


import google.generativeai as genai
import streamlit as st

def get_total_insight(data_summary, search_query):
    """
    내부 데이터와 외부 트렌드를 명확히 분리하여 3단 리포트를 작성하는 AI 엔진
    """
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # 사용 가능한 모델 탐색 및 연결
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not available_models:
            return ["오운완", "애슬레저", "운동하는여자"], "🚨 접근 가능한 모델이 없습니다."

        target_model = None
        for pref in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro', 'models/gemini-pro']:
            if pref in available_models:
                target_model = pref.replace('models/', '')
                break
        if not target_model:
            target_model = available_models[0].replace('models/', '')

        model = genai.GenerativeModel(target_model)
        
        # 3단 구조 프롬프트
        prompt = f"""
        당신은 글로벌 애슬레저 브랜드의 시니어 마케팅 전략가입니다. 

        [1. 내부 데이터 현황]
        {data_summary}

        [2. 외부 실시간 트렌드 분석 요청]
        키워드: "{search_query}"

        [🔥 작성 가이드 및 출력 형식]
        반드시 응답의 **첫 번째 줄**에는 트렌드를 반영하는 인스타그램 해시태그 2개만 '#' 없이 쉼표로 구분해 작성하세요. (예: 고프코어, 오운완)
        **두 번째 줄부터** 아래의 3단 구조 양식에 맞춰 마크다운 리포트를 작성하세요.

        ### 📊 Part 1. 내부 리뷰 데이터 (Voice of Customer)
        * **핵심 강점 (Strength):** (리뷰 데이터 기반 긍정 요소)
        * **주요 페인포인트 (Weakness):** (고객 불만 및 개선점)
        * **주요 언급 망 (네트워크):** (리뷰 내 주요 연결망 키워드 해석)

        ### 📸 Part 2. 외부 소셜 트렌드 (Instagram Real-time)
        * **현재 유행하는 스타일링:** (검색 키워드 관련 최신 트렌드)
        * **주목받는 신규 TPO:** (외부 시장의 기회 요소)

        ### 🚀 Part 3. 통합 전략 (Action Plan)
        > **💡 Strategic Gap:** (내부 데이터와 외부 유행 간의 차이점 및 시너지 포인트)
        * **마케팅 액션 아이템:** (구체적인 캠페인 제안)
        * **제품/UX 개선 아이템:** (구체적인 개선 제안)
        """
        
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        
        raw_hashtags = lines[0].strip().replace("#", "").split(',')
        dynamic_hashtags = [tag.strip() for tag in raw_hashtags]
        
        full_report = f"*(분석 모델: `{target_model}`)*\n\n" + '\n'.join(lines[1:])
        
        return dynamic_hashtags, full_report 

    except Exception as e:
        return ["레깅스추천", "애슬레저", "운동복코디"], f"🚨 인사이트 생성 중 오류 발생: {str(e)}"