import os, json, requests

def summarize_decision(context:dict)->str|None:
    api_key=os.getenv('OPENAI_API_KEY')
    if not api_key: return None
    url=os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1/chat/completions'); model=os.getenv('OPENAI_MODEL','gpt-4o-mini')
    headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'}
    messages=[{'role':'system','content':'Be concise. <=3 sentences.'},{'role':'user','content':json.dumps(context,default=str)}]
    payload={'model':model,'messages':messages,'temperature':0.2,'max_tokens':200}
    try:
        r=requests.post(url,headers=headers,json=payload,timeout=20); r.raise_for_status(); return r.json()['choices'][0]['message']['content']
    except Exception:
        return None
