

SYSTEM_TEMPLATE = """Sen araç kullanabilen bir ReAct ajanısın. Kullanıcının sorusunu \
yanıtlamak için aşağıdaki araçları kullanabilirsin:

{tools}

Yanıtını her zaman şu alanlarla ver ve YALNIZCA bu alanları üret:

Thought: (tek cümlelik düşünce)
Action: {tool_names} araçlarından birinin adı
Action Input: araca verilecek girdi

Her Action'dan sonra sistem sana bir "Observation:" satırı verir. Gerektiği kadar
Thought/Action/Observation döngüsü kurabilirsin. Cevabı bulduğunda şöyle bitir:

Thought: (tek cümlelik düşünce)
Final Answer: kullanıcıya verilecek nihai cevap

Kurallar:
- Başlık, madde imi, numaralı liste ya da meta-açıklama YAZMA; sadece yukarıdaki alanları üret.
- "Thought" en fazla bir cümle olsun; planını uzun uzun açıklama.
- "Observation:" satırını sen yazma; onu sistem ekler.
- Bir araca ihtiyacın yoksa doğrudan Thought + Final Answer yaz.
- "Final Answer:" yazdıktan sonra başka HİÇBİR ŞEY yazma.
- Kullanıcının diliyle (Türkçe) yanıtla."""

HISTORY_TEMPLATE = """Önceki konuşma (bağlam olarak kullan, gerekmiyorsa görmezden gel):
{turns}

"""


def _render_history(history):
    """history: [(soru, cevap), ...] -> user mesajına eklenecek bağlam metni."""
    if not history:
        return ""
    turns = "\n".join(f"Kullanıcı: {q}\nAsistan: {a}" for q, a in history)
    return HISTORY_TEMPLATE.format(turns=turns)


def build_system_prompt(tools_text, tool_names):
    """Sabit talimat bloğu — `system` rolünde gönderilir."""
    return SYSTEM_TEMPLATE.format(tools=tools_text, tool_names=tool_names)


def build_user_prompt(question, history, scratchpad):
    """Soru + geçmiş + o ana kadarki çalışma izi — `user` rolünde gönderilir."""
    parts = [_render_history(history), f"Question: {question}"]
    if scratchpad:
        parts.append(scratchpad)
    return "\n".join(p for p in parts if p)
