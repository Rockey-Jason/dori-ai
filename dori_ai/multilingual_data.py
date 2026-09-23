from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'generated'/'multilingual_foundation.jsonl'
FACTS={
'ko':[("돌이는 뭐야?","돌이는 오로라와 미요니의 웰시코기 인형 캐릭터야."),("돌이 키는 얼마야?","돌이는 약 17cm 정도의 작은 인형으로 설정되어 있어."),("돌이 AI는 무엇을 해?","돌이 AI는 질문에 답하고 돌이사이트 정보를 확인하도록 만든 AI야.")],
'en':[("What is Dori?","Dori is a Welsh Corgi plush character associated with Aurora and Miyo-ni."),("How tall is Dori?","Dori is described as a small plush, about 17 cm tall."),("What can Dori AI do?","Dori AI answers questions and can check information from the Dori site.")],
'ja':[("ドリとは何？","ドリはオーロラとミヨニに関連するウェルシュ・コーギーのぬいぐるみキャラクターだよ。"),("ドリの大きさは？","ドリは約17cmの小さなぬいぐるみとして設定されているよ。"),("ドリAIは何ができる？","ドリAIは質問に答え、ドリサイトの情報を確認できるよ。")],
'zh':[("朵莉是什么？","朵莉是与Aurora和Miyo-ni相关的威尔士柯基毛绒玩偶角色。"),("朵莉有多高？","朵莉设定为大约17厘米高的小型毛绒玩偶。"),("朵莉AI能做什么？","朵莉AI可以回答问题并查询朵莉网站的信息。")],
'es':[("¿Qué es Dori?","Dori es un personaje de peluche de Corgi galés asociado con Aurora y Miyo-ni."),("¿Cuánto mide Dori?","Dori está descrito como un pequeño peluche de unos 17 cm."),("¿Qué puede hacer Dori AI?","Dori AI responde preguntas y puede consultar información del sitio de Dori.")],
'fr':[("Qu'est-ce que Dori ?","Dori est un personnage en peluche de Corgi gallois associé à Aurora et Miyo-ni."),("Quelle est la taille de Dori ?","Dori est décrit comme une petite peluche d'environ 17 cm."),("Que peut faire Dori AI ?","Dori AI répond aux questions et peut consulter les informations du site Dori.")],
'de':[("Was ist Dori?","Dori ist eine Welsh-Corgi-Plüschfigur, die mit Aurora und Miyo-ni verbunden ist."),("Wie groß ist Dori?","Dori wird als kleines Plüschtier mit etwa 17 cm beschrieben."),("Was kann Dori AI?","Dori AI beantwortet Fragen und kann Informationen der Dori-Website prüfen.")],
'it':[("Che cos'è Dori?","Dori è un personaggio di peluche Welsh Corgi associato ad Aurora e Miyo-ni."),("Quanto è grande Dori?","Dori è descritto come un piccolo peluche di circa 17 cm."),("Cosa può fare Dori AI?","Dori AI risponde alle domande e può consultare le informazioni del sito Dori.")],
'pt':[("O que é Dori?","Dori é um personagem de pelúcia Welsh Corgi associado à Aurora e Miyo-ni."),("Qual é o tamanho de Dori?","Dori é descrito como um pequeno pelúcia de cerca de 17 cm."),("O que o Dori AI pode fazer?","Dori AI responde perguntas e pode consultar informações do site Dori.")],
'ru':[("Что такое Дори?","Дори — персонаж плюшевой игрушки в виде вельш-корги, связанный с Aurora и Miyo-ni."),("Какого размера Дори?","Дори описывается как небольшая игрушка высотой около 17 см."),("Что умеет Dori AI?","Dori AI отвечает на вопросы и может проверять информацию с сайта Dori.")],
}
rows=[]
for lang,pairs in FACTS.items():
    for q,a in pairs:
        for prefix in ("", "Please answer: ", "Question: ", "Answer this: ", "Tell me: ") if lang=='en' else ("",):
            rows.append({'question':prefix+q,'answer':a,'language':lang})
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
print(len(rows))
